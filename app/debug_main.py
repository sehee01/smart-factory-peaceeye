import cv2
import numpy as np
from pathlib import Path
import sys
import torch
from scipy.spatial.distance import cdist
import threading
import queue
import argparse
import time
from performance_logger import PerformanceLogger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRA_PATHS = [
    str(PROJECT_ROOT),                      # 루트 자체 (app, ByteTrack, frontend 등 import 가능)
    str(PROJECT_ROOT / "ByteTrack"),        # ByteTrack 직접 참조가 필요한 경우
    str(PROJECT_ROOT / "deep-person-reid-master"),  # 필요한 경우만
    str(PROJECT_ROOT / "app" / "models" / "mapping"),   # point_transformer 경로 수정
]

for p in EXTRA_PATHS:
    if p not in sys.path:
        sys.path.insert(0, p)

# np.float 호환성 문제 해결
if not hasattr(np, 'float'):
    np.float = float

from detector.detector_manager import ByteTrackDetectorManager
from reid.reid_manager import GlobalReIDManager
from reid.redis_handler import FeatureStoreRedisHandler
from reid.similarity import FeatureSimilarityCalculator
from config import settings
from models.mapping.point_transformer import transform_point
from image_processor import ImageProcessor


class AppOrchestrator:
    """
    전체 객체 탐지, 추적, ReID 재부여 흐름을 통합 실행하는 오케스트레이터
    멀티스레딩 지원으로 여러 비디오를 동시에 처리
    """

    def __init__(self, model_path: str, tracker_config: dict, redis_conf: dict, reid_conf: dict):
        # 공유 컴포넌트들
        self.model_path = model_path
        self.tracker_config = tracker_config
        self.redis_conf = redis_conf
        self.reid_conf = reid_conf
        
        # 로컬 ID와 글로벌 ID 매핑 저장소
        self.local_to_global_mapping = {}
        
        # ImageProcessor 초기화 (feature 추출 책임 분리)
        self.image_processor = ImageProcessor()

        # Redis 및 ReID 매니저 초기화 (공유)
        redis_handler = FeatureStoreRedisHandler(
            redis_host=redis_conf.get("host", "localhost"),
            redis_port=redis_conf.get("port", 6379),
            feature_ttl=reid_conf.get("ttl", 300)
        )
        similarity = FeatureSimilarityCalculator()
        self.reid = GlobalReIDManager(redis_handler, similarity, similarity_threshold=reid_conf.get("threshold", 0.7))

        self.camera_id = redis_conf.get("camera_id", "cam01")
        
        # 스레드별 성능 로거 저장소
        self.thread_performance_loggers = {}

    def create_detector_for_thread(self):
        """스레드별 독립적인 detector 생성"""
        return ByteTrackDetectorManager(self.model_path, self.tracker_config)

    def run_video(self, video_path):
        """단일 비디오 처리 (기존 메서드)"""
        detector = self.create_detector_for_thread()
        
        # 단일 비디오용 성능 로거 생성
        video_logger = PerformanceLogger(output_dir="result/single_video")
        self.thread_performance_loggers[0] = video_logger
        
        cap = cv2.VideoCapture(video_path)
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            
            # 성능 측정 시작
            video_logger.start_frame_timing(frame_id, 0)  # 단일 비디오는 카메라 ID 0
            
            # 글로벌 ReID 매니저 프레임 업데이트
            self.reid.update_frame(frame_id)
            
            # 탐지 및 트래킹 타이밍 시작
            video_logger.start_detection_timing()
            video_logger.start_tracking_timing()
            
            # 원본의 복잡한 탐지 로직 사용
            track_list = detector.detect_and_track(frame, frame_id)
            
            # 탐지 및 트래킹 타이밍 종료
            video_logger.end_detection_timing()
            video_logger.end_tracking_timing()

            # 프레임별 매칭된 트랙 추적
            frame_matched_tracks = set()
            
            # 객체 수 설정
            video_logger.set_object_count(len(track_list))

            for track in track_list:
                local_id = track["track_id"]
                bbox = track["bbox"]

                # --- Feature 추출 로직 (ImageProcessor 사용) ---
                crop, feature = self.image_processor.process_track_for_reid(frame, track)

                # 로컬 ID와 글로벌 ID 매핑 확인
                camera_key = f"{self.camera_id}_{local_id}"
                if camera_key in self.local_to_global_mapping:
                    # 기존 매핑이 있으면 사용 (같은 카메라 내 ReID)
                    global_id = self.local_to_global_mapping[camera_key]
                    print(f"[DEBUG] Using existing mapping: Local {local_id} -> Global {global_id}")
                    
                    # 같은 카메라 내 ReID 타이밍 시작
                    video_logger.start_same_camera_reid_timing()
                    
                    # ReID 매니저를 통해 업데이트 (Redis 상태 일관성 유지)
                    self.reid._update_track_camera(
                        global_id, feature, bbox, self.camera_id, frame_id, local_id
                    )
                    
                    # 같은 카메라 내 ReID 타이밍 종료
                    video_logger.end_same_camera_reid_timing()
                else:
                    # 새로운 ReID 매칭 시도 (다른 카메라 간 ReID)
                    video_logger.start_cross_camera_reid_timing()
                    
                    global_id = self.reid.match_or_create(
                        features=feature,
                        bbox=bbox,
                        camera_id=self.camera_id,
                        frame_id=frame_id,
                        frame_shape=frame.shape[:2],
                        matched_tracks=frame_matched_tracks,  # 프레임 내에서 공유
                        local_track_id=local_id
                    )
                    
                    # 다른 카메라 간 ReID 타이밍 종료
                    video_logger.end_cross_camera_reid_timing()
                    
                    if global_id is None:
                        global_id = local_id  # 매칭 실패 시 로컬 ID 사용
                    else:
                        # 새로운 매핑 저장
                        self.local_to_global_mapping[camera_key] = global_id
                        print(f"[DEBUG] New mapping: Local {local_id} -> Global {global_id}")

                # --- 좌표 변환 (ImageProcessor 사용) ---
                x1, y1, x2, y2, point_x, point_y = self.image_processor.get_bbox_coordinates(bbox)
                
                try:
                    # 실제 좌표로 변환 (설정에서 가져온 매트릭스 사용)
                    real_x, real_y = point_x, point_y #테스트시 연산 최소화 위한 옵션
                    # real_x, real_y = transform_point(point_x, point_y, settings.HOMOGRAPHY_MATRIX)
                    
                    print(f"[DEBUG] Camera {self.camera_id}, Worker {global_id}: Image({point_x:.1f}, {point_y:.1f}) -> Real({real_x:.4f}, {real_y:.4f})")
                except Exception as e:
                    print(f"Warning: Coordinate transformation failed: {e}")
                    real_x, real_y = point_x, point_y

                # --- 결과 디스플레이 ---
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {global_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 성능 데이터 로깅
            video_logger.log_frame_performance()

            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def run_video_thread(self, video_path, camera_id, frame_buffer, sync_event, stop_event):
        """프레임 수집 전용 스레드 (동기화됨)"""
        detector = self.create_detector_for_thread()
        
        # 스레드별 독립적인 성능 로거 생성
        thread_logger = PerformanceLogger(output_dir=f"result/camera_{camera_id}")
        self.thread_performance_loggers[camera_id] = thread_logger
        
        cap = cv2.VideoCapture(video_path)
        frame_id = 0  # 원본 비디오의 실제 프레임 번호
        processed_frame_id = 0  # 실제 처리된 프레임의 순차적 번호

        # 프레임 스킵 설정
        target_fps = 30  # 목표 FPS
        frame_skip = 30/target_fps # 스킵할 프레임 수
        
        while cap.isOpened() and not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print(f"[{video_path}] Video ended")
                break

            frame_id += 1  # 원본 프레임 번호 증가

            # 프레임 스킵: frame_skip 프레임마다 하나씩만 처리
            if frame_id % frame_skip != 0:
                print(f"[DEBUG] Skipping frame {frame_id}")
                continue  # 이 프레임은 스킵
            
            processed_frame_id += 1  # 처리된 프레임 번호 증가
            print(f"[DEBUG] Processing frame {frame_id} -> processed_frame_id {processed_frame_id}")
            
            # 성능 측정 시작
            thread_logger.start_frame_timing(processed_frame_id, camera_id)
            
            # 글로벌 ReID 매니저 프레임 업데이트 (processed_frame_id 사용)
            self.reid.update_frame(processed_frame_id)
            
            # 탐지 및 트래킹 타이밍 시작
            thread_logger.start_detection_timing()
            thread_logger.start_tracking_timing()
            
            # 원본의 복잡한 탐지 로직 사용 (processed_frame_id 사용)
            track_list = detector.detect_and_track(frame, processed_frame_id)
            
            # 탐지 및 트래킹 타이밍 종료
            thread_logger.end_detection_timing()
            thread_logger.end_tracking_timing()

            # 프레임별 매칭된 트랙 추적
            frame_matched_tracks = set()
            
            frame_detections_json = []
            
            # 객체 수 설정
            thread_logger.set_object_count(len(track_list))
            
            for track in track_list:
                local_id = track["track_id"]
                bbox = track["bbox"]

                # --- Feature 추출 로직 (ImageProcessor 사용) ---
                crop, feature = self.image_processor.process_track_for_reid(frame, track)

                # 로컬 ID와 글로벌 ID 매핑 확인
                camera_key = f"{camera_id}_{local_id}"
                if camera_key in self.local_to_global_mapping:
                    # 기존 매핑이 있으면 사용 (같은 카메라 내 ReID)
                    global_id = self.local_to_global_mapping[camera_key]
                    
                    print(f"[DEBUG] Using existing mapping: Local {local_id} -> Global {global_id}")
                    
                    # 같은 카메라 내 ReID 타이밍 시작
                    thread_logger.start_same_camera_reid_timing()
                    
                    # ReID 매니저를 통해 업데이트 (frame_id 사용 - 원본 프레임 번호)
                    self.reid._update_track_camera(
                        global_id, feature, bbox, str(camera_id), frame_id, local_id
                    )
                    
                    # 같은 카메라 내 ReID 타이밍 종료
                    thread_logger.end_same_camera_reid_timing()
                else:
                    # 새로운 ReID 매칭 시도 (다른 카메라 간 ReID)
                    thread_logger.start_cross_camera_reid_timing()
                    
                    global_id = self.reid.match_or_create(
                        features=feature,
                        bbox=bbox,
                        camera_id=str(camera_id),
                        frame_id=frame_id,  # 원본 프레임 번호
                        frame_shape=frame.shape[:2],
                        matched_tracks=frame_matched_tracks,
                        local_track_id=local_id
                    )
                    
                    # 다른 카메라 간 ReID 타이밍 종료
                    thread_logger.end_cross_camera_reid_timing()
                    
                    if global_id is None:
                        global_id = local_id
                    else:
                        # 새로운 매핑 저장 reid 매칭에 성공
                        self.local_to_global_mapping[camera_key] = global_id
                        print(f"[DEBUG] New mapping: Local {local_id} -> Global {global_id}")

                # --- 좌표 변환 ---
                x1, y1, x2, y2, point_x, point_y = self.image_processor.get_bbox_coordinates(bbox)
                
                try:
                    real_x, real_y = transform_point(point_x, point_y, settings.HOMOGRAPHY_MATRIX)
                    print(f"[DEBUG] Camera {camera_id}, Worker {global_id}: Image({point_x:.1f}, {point_y:.1f}) -> Real({real_x:.4f}, {real_y:.4f})")
                except Exception as e:
                    print(f"Warning: Coordinate transformation failed: {e}")
                    real_x, real_y = point_x, point_y

                # JSON 데이터 생성 (processed_frame_id 사용 - 처리된 프레임 번호)
                detection_data = {
                    "cameraID": int(camera_id),
                    "workerID": int(global_id),
                    "position_X": real_x,
                    "position_Y": real_y,
                    "frame_id": processed_frame_id  # 처리된 프레임 번호
                }
                frame_detections_json.append(detection_data)

                # 바운딩 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID:{global_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 성능 데이터 로깅
            thread_logger.log_frame_performance()
            
            # 프레임 데이터를 버퍼에 저장 (processed_frame_id 사용)
            frame_data = {
                'video_path': video_path,
                'frame': frame,
                'detections': frame_detections_json,
                'frame_id': processed_frame_id  # 처리된 프레임 번호
            }
            
            # 동기화: 메인 스레드가 처리할 준비가 될 때까지 대기
            frame_buffer.put(frame_data)
            
            # 메인 스레드의 처리 완료 신호 대기
            sync_event.wait(0.1)
            sync_event.clear()  # 이벤트 리셋

        cap.release()
        # 종료 신호 전송
        frame_buffer.put({'video_path': video_path, 'frame': None, 'detections': None, 'frame_id': -1}) 

    def run_multi_video(self, video_paths):
        """동기화된 멀티 비디오 처리"""
        stop_event = threading.Event()
        sync_event = threading.Event()  # 동기화 이벤트
        
        # 각 비디오별 프레임 버퍼
        frame_buffers = {video_path: queue.Queue(maxsize=1) for video_path in video_paths}
        
        # 프레임 수집 스레드 생성
        collector_threads = []
        for i, video_path in enumerate(video_paths):
            thread = threading.Thread(
                target=self.run_video_thread,
                args=(video_path, i, frame_buffers[video_path], sync_event, stop_event),
                daemon=True
            )
            collector_threads.append(thread)
            thread.start()

        # GUI 처리
        latest_frames = {}
        active_videos = set(video_paths)
        latest_detections = {}
        
        # 각 비디오별 창 생성
        window_names = {}
        for i, video_path in enumerate(video_paths):
            window_name = f"Camera {i} - {Path(video_path).name}"
            window_names[video_path] = window_name
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, settings.GUI_CONFIG["window_width"], settings.GUI_CONFIG["window_height"])
        
        # 동기화된 프레임 처리
        frame_count = 0
        max_frames = settings.MULTITHREADING_CONFIG["max_frames"]
        
        while active_videos and frame_count < max_frames:
            # 모든 비디오에서 프레임이 준비될 때까지 대기
            all_frames_ready = True
            current_frames = {}
            
            for video_path in active_videos.copy():
                try:
                    frame_data = frame_buffers[video_path].get(timeout=0.01)  # 10ms 타임아웃
                    
                    if frame_data['frame'] is None:
                        # 비디오 종료 신호
                        active_videos.discard(video_path)
                        if video_path in latest_detections:
                            del latest_detections[video_path]
                        continue
                    
                    current_frames[video_path] = frame_data
                    
                except queue.Empty:
                    # 프레임이 아직 준비되지 않음
                    all_frames_ready = False
                    break
            
            # 모든 프레임이 준비되지 않았으면 잠시 대기
            if not all_frames_ready:
                time.sleep(0.01)  # 10ms 대기
                continue
            
            # 모든 프레임이 준비되었으면 동시에 처리
            frame_count += 1
            print(f"[SYNC] Processing frame {frame_count} for all videos")
            
            for video_path, frame_data in current_frames.items():
                frame = frame_data['frame']
                detections = frame_data['detections']
                frame_id = frame_data['frame_id']
                
                if detections:
                    latest_detections[video_path] = detections

                # 각 비디오를 별도의 창에 표시
                if frame is not None:
                    window_name = window_names[video_path]
                    cv2.imshow(window_name, frame)
            
            # 모든 창에서 'q' 키를 누르면 종료
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # 모든 스레드에게 다음 프레임 처리 허가 신호 전송
            sync_event.set()
        
        # 모든 카메라의 감지 결과를 합쳐서 반환
        all_detections = []
        for detections in latest_detections.values():
            all_detections.extend(detections)
        
        # 스레드 정리
        stop_event.set()
        for thread in collector_threads:
            thread.join(timeout=settings.MULTITHREADING_CONFIG["thread_timeout"])
        
        # 모든 창 닫기
        for window_name in window_names.values():
            cv2.destroyWindow(window_name)
        
        return all_detections




def main():
    """메인 함수: 멀티스레딩 지원"""
    import argparse
    from config import settings

    parser = argparse.ArgumentParser(
        description="YOLOv8 with ByteTrack and Redis Global Re-ID V2 for Multi-Video Tracking"
    )
    parser.add_argument(
        '--videos',
        nargs='+',
        type=str,
        default=settings.VIDEO_INPUT_PATHS,
        help='List of video file paths.'
    )
    parser.add_argument(
        '--yolo_model',
        type=str,
        default="models/weights/bestcctv.pt",
        help='Path to the YOLOv8 model file.'
    )
    parser.add_argument(
        '--redis_host',
        type=str,
        default="localhost",
        help='Redis server host.'
    )
    parser.add_argument(
        '--redis_port',
        type=int,
        default=6379,
        help='Redis server port.'
    )
    parser.add_argument(
        '--multi_thread',
        action='store_true',
        help='Enable multi-threading for multiple videos'
    )
    args = parser.parse_args()

    tracker_config = settings.TRACKER_CONFIG
    reid_config = settings.REID_CONFIG

    redis_conf = {
        "host": args.redis_host,
        "port": args.redis_port,
        "camera_id": "cam01"
    }

    app = AppOrchestrator(
        model_path=args.yolo_model,
        tracker_config=tracker_config,
        redis_conf=redis_conf,
        reid_conf=reid_config
    )

    if args.multi_thread and len(args.videos) > 1:
        print(f"▶ Processing {len(args.videos)} videos with multi-threading")
        for video_path in args.videos:
            print(f"  - {video_path}")
        
        # 멀티스레딩으로 여러 비디오 처리
        all_detections = app.run_multi_video(args.videos)
        print(f"▶ Total detections: {len(all_detections)}")
        
        # 성능 요약 출력 (모든 스레드 로거 합쳐서)
        print("\n" + "="*60)
        print("📊 COMBINED PERFORMANCE SUMMARY")
        print("="*60)
        
        # 각 스레드별 로거 요약
        for camera_id, thread_logger in app.thread_performance_loggers.items():
            print(f"\n📊 Camera {camera_id} Performance Summary:")
            thread_logger.print_summary()
        
        print("="*60)
    else:
        # 단일 비디오 처리 (기존 방식)
        for video_path in args.videos:
            print(f"▶ Processing video: {video_path}")
            app.run_video(video_path)
        
        # 성능 요약 출력 (단일 비디오의 경우)
        if app.thread_performance_loggers:
            for camera_id, thread_logger in app.thread_performance_loggers.items():
                print(f"\n📊 Camera {camera_id} Performance Summary:")
                thread_logger.print_summary()
        else:
            print("📊 No performance data available")


if __name__ == '__main__':
    main()
