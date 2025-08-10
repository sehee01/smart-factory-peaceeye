import cv2
import numpy as np
from pathlib import Path
import sys
import torch
from scipy.spatial.distance import cdist
import threading
import queue
import argparse

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

from torchreid.utils.feature_extractor import FeatureExtractor
from detector.detector_manager import ByteTrackDetectorManager
from reid.reid_manager import GlobalReIDManager
from reid.redis_handler import FeatureStoreRedisHandler
from reid.similarity import FeatureSimilarityCalculator
from config import settings
from models.mapping.point_transformer import transform_point


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
        
        # Feature Extractor 초기화 (설정에서 가져온 값 사용)
        device = settings.FEATURE_EXTRACTOR_CONFIG["device"]
        if device == "auto":
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.feature_extractor = FeatureExtractor(
            model_name=settings.FEATURE_EXTRACTOR_CONFIG["model_name"],
            model_path=settings.FEATURE_EXTRACTOR_CONFIG["model_path"],
            device=device
        )

        # Redis 및 ReID 매니저 초기화 (공유)
        redis_handler = FeatureStoreRedisHandler(
            redis_host=redis_conf.get("host", "localhost"),
            redis_port=redis_conf.get("port", 6379),
            feature_ttl=reid_conf.get("ttl", 300)
        )
        similarity = FeatureSimilarityCalculator()
        self.reid = GlobalReIDManager(redis_handler, similarity, similarity_threshold=reid_conf.get("threshold", 0.7))

        self.camera_id = redis_conf.get("camera_id", "cam01")

    def create_detector_for_thread(self):
        """스레드별 독립적인 detector 생성"""
        return ByteTrackDetectorManager(self.model_path, self.tracker_config)

    def run_video(self, video_path):
        """단일 비디오 처리 (기존 메서드)"""
        detector = self.create_detector_for_thread()
        cap = cv2.VideoCapture(video_path)
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            
            # 글로벌 ReID 매니저 프레임 업데이트
            self.reid.update_frame(frame_id)
            
            # 원본의 복잡한 탐지 로직 사용
            track_list = detector.detect_and_track(frame, frame_id)

            # 프레임별 매칭된 트랙 추적
            frame_matched_tracks = set()

            for track in track_list:
                local_id = track["track_id"]
                bbox = track["bbox"]

                # --- Feature 추출 로직 ---
                x1, y1, x2, y2 = map(int, bbox)
                crop = frame[y1:y2, x1:x2]
                
                if self.feature_extractor is not None and crop.size > 0:
                    # 전문적인 feature extractor 사용
                    feature = self._extract_feature_with_extractor(crop)
                else:
                    # 단순한 RGB 평균 feature (fallback)
                    feature = self._extract_feature_simple(crop)

                # 로컬 ID와 글로벌 ID 매핑 확인
                camera_key = f"{self.camera_id}_{local_id}"
                if camera_key in self.local_to_global_mapping:
                    # 기존 매핑이 있으면 사용
                    global_id = self.local_to_global_mapping[camera_key]
                    print(f"[DEBUG] Using existing mapping: Local {local_id} -> Global {global_id}")
                    # 기존 매핑이 있어도 feature는 계속 저장 (문제 해결)
                    self.reid.redis.store_feature_with_metadata(
                        global_id, str(self.camera_id), frame_id, feature, bbox, 
                        self.reid.global_frame_counter, local_id
                    )
                else:
                    # 새로운 ReID 매칭 시도
                    global_id = self.reid.match_or_create(
                        features=feature,
                        bbox=bbox,
                        camera_id=self.camera_id,
                        frame_id=frame_id,
                        frame_shape=frame.shape[:2],
                        matched_tracks=frame_matched_tracks,  # 프레임 내에서 공유
                        local_track_id=local_id
                    )
                    
                    if global_id is None:
                        global_id = local_id  # 매칭 실패 시 로컬 ID 사용
                    else:
                        # 새로운 매핑 저장
                        self.local_to_global_mapping[camera_key] = global_id
                        print(f"[DEBUG] New mapping: Local {local_id} -> Global {global_id}")

                # --- 좌표 변환 (원본 기능 복원) ---
                point_x = (x1 + x2) / 2
                point_y = y1
                
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

            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def run_video_thread(self, video_path, camera_id, frame_queue, stop_event):
        """스레드용 비디오 처리 함수"""
        # 스레드별 독립적인 detector 생성
        detector = self.create_detector_for_thread()
        
        cap = cv2.VideoCapture(video_path)
        frame_id = 0
        person_count = 0
        tracked_ids = set()

        while cap.isOpened() and not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print(f"[{video_path}] Video ended")
                break

            frame_id += 1
            
            # 글로벌 ReID 매니저 프레임 업데이트
            self.reid.update_frame(frame_id)
            
            # 원본의 복잡한 탐지 로직 사용
            track_list = detector.detect_and_track(frame, frame_id)

            # 프레임별 매칭된 트랙 추적
            frame_matched_tracks = set()
            
            frame_detections_json = []
            for track in track_list:
                local_id = track["track_id"]
                bbox = track["bbox"]

                if local_id not in tracked_ids:
                    tracked_ids.add(local_id)
                    person_count += 1

                # --- Feature 추출 로직 ---
                x1, y1, x2, y2 = map(int, bbox)
                crop = frame[y1:y2, x1:x2]
                
                if self.feature_extractor is not None and crop.size > 0:
                    feature = self._extract_feature_with_extractor(crop)
                else:
                    feature = self._extract_feature_simple(crop)

                # 로컬 ID와 글로벌 ID 매핑 확인
                camera_key = f"{camera_id}_{local_id}"
                if camera_key in self.local_to_global_mapping:
                    # 기존 매핑이 있으면 사용
                    global_id = self.local_to_global_mapping[camera_key]
                    
                    print(f"[DEBUG] Using existing mapping: Local {local_id} -> Global {global_id}")
                    
                    # 기존 매핑이 있어도 feature는 계속 저장 (문제 해결)
                    self.reid.redis.store_feature_with_metadata(
                        global_id, str(camera_id), frame_id, feature, bbox, 
                        self.reid.global_frame_counter, local_id
                    )
                else:
                    # 새로운 ReID 매칭 시도
                    global_id = self.reid.match_or_create(
                        features=feature,
                        bbox=bbox,
                        camera_id=str(camera_id),
                        frame_id=frame_id,
                        frame_shape=frame.shape[:2],
                        matched_tracks=frame_matched_tracks,  # 프레임 내에서 공유
                        local_track_id=local_id
                    )
                    
                    if global_id is None:
                        global_id = local_id # 수정 필요요
                    else:
                        # 새로운 매핑 저장
                        self.local_to_global_mapping[camera_key] = global_id
                        print(f"[DEBUG] New mapping: Local {local_id} -> Global {global_id}")

                # --- 좌표 변환 ---
                point_x = (x1 + x2) / 2
                point_y = y1
                
                try:
                    real_x, real_y = transform_point(point_x, point_y, settings.HOMOGRAPHY_MATRIX)
                    print(f"[DEBUG] Camera {camera_id}, Worker {global_id}: Image({point_x:.1f}, {point_y:.1f}) -> Real({real_x:.4f}, {real_y:.4f})")
                except Exception as e:
                    print(f"Warning: Coordinate transformation failed: {e}")
                    real_x, real_y = point_x, point_y

                # JSON 데이터 생성
                detection_data = {
                    "cameraID": int(camera_id),
                    "workerID": int(global_id),
                    "position_X": real_x,
                    "position_Y": real_y,
                    "frame_id": frame_id
                }
                frame_detections_json.append(detection_data)

                # 바운딩 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID:{global_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 프레임과 감지 결과를 큐에 전송
            frame_queue.put((video_path, frame, frame_detections_json))

        cap.release()
        frame_queue.put((video_path, None, None))

    def run_multi_video(self, video_paths):
        """여러 비디오를 동시에 처리하는 멀티스레딩 함수"""
        frame_queue = queue.Queue()
        stop_event = threading.Event()
        
        # 워커 스레드 생성
        threads = []
        for i, video_path in enumerate(video_paths):
            thread = threading.Thread(
                target=self.run_video_thread,
                args=(video_path, i, frame_queue, stop_event),
                daemon=True
            )
            threads.append(thread)
            thread.start()

        # GUI 처리
        latest_frames = {}
        active_videos = set(video_paths)
        latest_detections = {}
        
        # 각 비디오별 창 생성 (설정에서 가져온 값 사용)
        window_names = {}
        for i, video_path in enumerate(video_paths):
            window_name = f"Camera {i} - {Path(video_path).name}"
            window_names[video_path] = window_name
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, settings.GUI_CONFIG["window_width"], settings.GUI_CONFIG["window_height"])
        
        # 매 프레임마다 실시간 처리 (설정에서 가져온 값 사용)
        frame_count = 0
        max_frames = settings.MULTITHREADING_CONFIG["max_frames"]
        
        while active_videos and frame_count < max_frames:
            try:
                video_path, frame, detections = frame_queue.get(timeout=settings.MULTITHREADING_CONFIG["frame_timeout"])

                if frame is None:
                    active_videos.discard(video_path)
                    if video_path in latest_detections:
                        del latest_detections[video_path]
                    continue
                
                frame_count += 1
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

            except queue.Empty:
                continue
        
        # 모든 카메라의 감지 결과를 합쳐서 반환
        all_detections = []
        for detections in latest_detections.values():
            all_detections.extend(detections)
        
        # 스레드 정리 (설정에서 가져온 값 사용)
        stop_event.set()
        for thread in threads:
            thread.join(timeout=settings.MULTITHREADING_CONFIG["thread_timeout"])
        
        # 모든 창 닫기
        for window_name in window_names.values():
            cv2.destroyWindow(window_name)
        
        return all_detections

    def _extract_feature_with_extractor(self, crop_img):
        """전문적인 feature extractor를 사용한 feature 추출 (패딩 없이 resize만 사용)"""
        if crop_img.size == 0:
            return np.zeros(512)  # osnet_ibn_x1_0의 feature dimension
        
        # 패딩 없이 단순 resize
        target_size = settings.FEATURE_EXTRACTOR_CONFIG["target_size"]
        resized_crop = cv2.resize(crop_img, target_size)
        normalized_crop = resized_crop.astype(np.float32) / 255.0
        
        # Feature 추출
        with torch.no_grad():
            feature = self.feature_extractor([normalized_crop]).cpu().numpy()
            return feature.flatten()

    def _extract_feature_simple(self, crop_img):
        """단순한 RGB 평균 feature 추출 (fallback)"""
        if crop_img.size == 0:
            return np.zeros(3)
        feature = crop_img.mean(axis=(0, 1))  # RGB 평균
        return feature / 255.0


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
    else:
        # 단일 비디오 처리 (기존 방식)
        for video_path in args.videos:
            print(f"▶ Processing video: {video_path}")
            app.run_video(video_path)


if __name__ == '__main__':
    main()
