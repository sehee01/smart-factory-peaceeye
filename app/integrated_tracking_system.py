import cv2
import numpy as np
import threading
import queue
import time
import json
from pathlib import Path

from detector.detector_manager import ByteTrackDetectorManager
from reid.reid_manager import GlobalReIDManager
from reid.redis_handler import FeatureStoreRedisHandler
from reid.similarity import FeatureSimilarityCalculator
from config import settings
from image_processor import ImageProcessor
from result.performance_logger import PerformanceLogger

from ppe_detector import PPEDetector
from homography_manager import HomographyManager
from backend_client import BackendClient
from models.mapping.point_transformer import transform_point


class IntegratedTrackingSystem:
    """
    ReID 기능과 호모그래피 기능, PPE 탐지 기능을 통합한 객체 추적 시스템
    두 개의 영상에서 객체 추적이 잘 되도록 호모그래피 좌표를 직접 입력하는 방식
    """
    
    def __init__(self, video_paths=None, model_path=None, tracker_config=None, 
                 redis_conf=None, reid_conf=None, calibration_files=None, backend_url=None,
                 ppe_model_path=None):
        # 설정 로드
        self.model_path = model_path or settings.YOLO_MODEL_PATH
        self.tracker_config = tracker_config or settings.TRACKER_CONFIG
        redis_conf = redis_conf or settings.REDIS_CONFIG
        reid_conf = reid_conf or settings.REID_CONFIG
        
        # 비디오 경로 설정
        self.video_paths = video_paths or settings.VIDEO_INPUT_PATHS
        
        # 호모그래피 매니저 초기화
        self.homography_manager = HomographyManager()
        
        # PPE 탐지기 초기화
        self.ppe_detector = PPEDetector(ppe_model_path or "models/weights/best_yolo11n.pt")
        
        # 캘리브레이션 파일 로드 (명령행 인자 우선, 없으면 settings에서 자동 로드)
        if calibration_files:
            for camera_id, calib_file in calibration_files.items():
                self.homography_manager.add_camera_calibration(camera_id, calib_file)
        else:
            # settings.py에서 자동으로 호모그래피 매트릭스 로드
            for camera_id, matrix in settings.HOMOGRAPHY_MATRICES.items():
                self.homography_manager.homography_matrices[camera_id] = np.array(matrix)
                print(f"Camera {camera_id} homography matrix loaded from settings")
        
        # 백엔드 클라이언트 초기화
        self.backend_client = BackendClient(backend_url or "http://localhost:5000")
        
        # Redis 핸들러 초기화
        redis_handler = FeatureStoreRedisHandler(
            redis_host=redis_conf.get("host", "localhost"),
            redis_port=redis_conf.get("port", 6379)
        )
        similarity = FeatureSimilarityCalculator()
        self.reid = GlobalReIDManager(redis_handler, similarity, similarity_threshold=reid_conf.get("threshold", 0.7))
        
        # 로컬 ID와 글로벌 ID 매핑 저장소
        self.local_to_global_mapping = {}
        
        # ImageProcessor 초기화
        self.image_processor = ImageProcessor()
        
        # 스레드별 성능 로거 저장소
        self.thread_performance_loggers = {}
        
        # 추적 결과 저장소
        self.tracking_results = {}
        
        # PPE 위반 추적 저장소 (중복 전송 방지)
        self.ppe_violation_history = {}
        
        print("Integrated Tracking System initialized")
        print(f"Videos: {self.video_paths}")
        print(f"Model: {self.model_path}")
        print(f"PPE Model: {ppe_model_path or 'models/weights/best_yolo11n.pt'}")
        print(f"Backend URL: {backend_url or 'http://localhost:5000'}")
        
        # 로드된 호모그래피 매트릭스 정보 출력
        print(f"Loaded homography matrices for cameras: {list(self.homography_manager.homography_matrices.keys())}")
        for camera_id in self.homography_manager.homography_matrices.keys():
            print(f"  Camera {camera_id}: Matrix shape {self.homography_manager.homography_matrices[camera_id].shape}")
    
    def create_detector_for_thread(self):
        """스레드별 독립적인 detector 생성"""
        return ByteTrackDetectorManager(self.model_path, self.tracker_config)
    
    def process_frame_with_reid_and_homography(self, frame, frame_id, camera_id, detector):
        """ReID와 호모그래피를 통합한 프레임 처리"""
        # 성능 측정 시작
        if camera_id not in self.thread_performance_loggers:
            self.thread_performance_loggers[camera_id] = PerformanceLogger(output_dir=f"result/camera_{camera_id}")
        
        logger = self.thread_performance_loggers[camera_id]
        logger.start_frame_timing(frame_id, camera_id)
        
        # 글로벌 ReID 매니저 프레임 업데이트
        self.reid.update_frame(frame_id)
        
        # 탐지 및 트래킹
        logger.start_detection_timing()
        logger.start_tracking_timing()
        
        track_list = detector.detect_and_track(frame, frame_id)
        
        logger.end_detection_timing()
        logger.end_tracking_timing()
        
        # 프레임별 매칭된 트랙 추적
        frame_matched_tracks = set()
        frame_detections = []
        frame_violations = []
        
        # 객체 수 설정
        logger.set_object_count(len(track_list))
        
        for track in track_list:
            local_id = track["track_id"]
            bbox = track["bbox"]
            
            # Feature 추출
            crop, feature = self.image_processor.process_track_for_reid(frame, track)
            
            # ReID 매칭
            camera_key = f"{camera_id}_{local_id}"
            if camera_key in self.local_to_global_mapping:
                # 기존 매핑 사용
                global_id = self.local_to_global_mapping[camera_key]
                logger.start_same_camera_reid_timing()
                
                self.reid._update_track_camera(
                    global_id, feature, bbox, str(camera_id), frame_id, local_id
                )
                
                logger.end_same_camera_reid_timing()
            else:
                # 새로운 ReID 매칭
                logger.start_cross_camera_reid_timing()
                
                global_id = self.reid.match_or_create(
                    features=feature,
                    bbox=bbox,
                    camera_id=str(camera_id),
                    frame_id=frame_id,
                    frame_shape=frame.shape[:2],
                    matched_tracks=frame_matched_tracks,
                    local_track_id=local_id
                )
                
                logger.end_cross_camera_reid_timing()
                
                if global_id is None:
                    global_id = local_id
                else:
                    self.local_to_global_mapping[camera_key] = global_id
            
            # 좌표 변환
            x1, y1, x2, y2, point_x, point_y = self.image_processor.get_bbox_coordinates(bbox)
            
            # 호모그래피 변환 (point_transformer 사용)
            try:
                homography_matrix = np.array(settings.HOMOGRAPHY_MATRICES[camera_id])
                real_x, real_y = transform_point(point_x, point_y, homography_matrix)
            except Exception as e:
                print(f"Error in coordinate transformation for camera {camera_id}: {e}")
                real_x, real_y = 0, 0
            
            # PPE 위반 탐지
            ppe_violations = self.ppe_detector.detect_ppe_violations(frame, bbox)
            
            # PPE 위반이 있으면 위반 데이터 생성
            if ppe_violations:
                violation_key = f"{camera_id}_{global_id}_{frame_id}"
                if violation_key not in self.ppe_violation_history:
                    self.ppe_violation_history[violation_key] = True
                    
                    violation_data = {
                        "worker_id": f"W{global_id:03d}",
                        "zone_id": f"Z{camera_id:02d}",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "violations": {
                            "ppe": [v['type'] for v in ppe_violations],
                            "roi": []
                        }
                    }
                    frame_violations.append(violation_data)
                    
                    print(f"[PPE VIOLATION] Worker {global_id} in Camera {camera_id}: {[v['type'] for v in ppe_violations]}")
            
            # 결과 저장
            detection_data = {
                "cameraID": int(camera_id),
                "workerID": int(global_id),
                "position_X": real_x,
                "position_Y": real_y,
                "frame_id": frame_id,
                "local_id": local_id,
                "bbox": [x1, y1, x2, y2],
                "image_coords": [point_x, point_y],
                "ppe_violations": [v['type'] for v in ppe_violations] if ppe_violations else []
            }
            frame_detections.append(detection_data)
            
            # 바운딩 박스 그리기
            color = (0, 0, 255) if ppe_violations else (0, 255, 0)  # 위반시 빨간색, 정상시 초록색
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'ID:{global_id}', (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # 좌표 정보 표시
            coord_text = f"({real_x:.1f}, {real_y:.1f})"
            cv2.putText(frame, coord_text, (x1, y2 + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # PPE 위반 정보 표시
            if ppe_violations:
                violation_text = f"PPE: {', '.join([v['type'] for v in ppe_violations])}"
                cv2.putText(frame, violation_text, (x1, y2 + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # 성능 데이터 로깅
        logger.log_frame_performance()
        
        return frame, frame_detections, frame_violations
    
    def send_detections_to_backend(self, detections, violations):
        """감지 결과와 위반 결과를 백엔드로 전송"""
        # 워커 데이터 전송
        if detections:
            worker_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "workers": []
            }
            
            for det in detections:
                # NumPy 타입을 Python 기본 타입으로 변환
                worker_info = {
                    "worker_id": str(det["workerID"]),
                    "zone_id": f"Z{det['cameraID']:02d}",
                    "x": float(det["position_X"]),  # float32를 float로 변환
                    "y": float(det["position_Y"]),  # float32를 float로 변환
                    "product_count": 0,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                worker_data["workers"].append(worker_info)
            
            # 백엔드로 전송
            self.backend_client.send_worker_data(worker_data)
        
        # 위반 데이터 전송
        if violations:
            violation_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "violations": []
            }
            
            for violation in violations:
                violation_entry = {
                    "worker_id": str(violation["worker_id"]),
                    "zone_id": str(violation["zone_id"]),
                    "timestamp": str(violation["timestamp"]),
                    "violations": violation["violations"]
                }
                violation_data["violations"].append(violation_entry)
            
            # 백엔드로 전송
            self.backend_client.send_violation_data(violation_data)
    
    def run_video_thread(self, video_path, camera_id, frame_buffer, sync_event, stop_event):
        """비디오 처리 스레드"""
        detector = self.create_detector_for_thread()
        
        cap = cv2.VideoCapture(video_path)
        frame_id = 0
        processed_frame_id = 0
        
        # 프레임 스킵 설정
        target_fps = 30
        frame_skip = 30/target_fps
        
        while cap.isOpened() and not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print(f"[{video_path}] Video ended")
                break
            
            frame_id += 1
            
            # 프레임 스킵
            if frame_id % frame_skip != 0:
                continue
            
            processed_frame_id += 1
            
            # 프레임 처리
            processed_frame, detections, violations = self.process_frame_with_reid_and_homography(
                frame, processed_frame_id, camera_id, detector
            )
            
            # 백엔드로 전송
            self.send_detections_to_backend(detections, violations)
            
            # 결과 저장
            frame_data = {
                'video_path': video_path,
                'frame': processed_frame,
                'detections': detections,
                'violations': violations,
                'frame_id': processed_frame_id
            }
            
            # 버퍼에 저장
            frame_buffer.put(frame_data)
            
            # 동기화 대기
            sync_event.wait(0.1)
            sync_event.clear()
        
        cap.release()
        frame_buffer.put({'video_path': video_path, 'frame': None, 'detections': None, 'violations': None, 'frame_id': -1})
    
    def run_multi_video_tracking(self):
        """멀티 비디오 통합 추적 실행"""
        stop_event = threading.Event()
        sync_event = threading.Event()
        
        # 각 비디오별 프레임 버퍼
        frame_buffers = {video_path: queue.Queue(maxsize=1) for video_path in self.video_paths}
        
        # 프레임 수집 스레드 생성
        collector_threads = []
        for i, video_path in enumerate(self.video_paths):
            thread = threading.Thread(
                target=self.run_video_thread,
                args=(video_path, i, frame_buffers[video_path], sync_event, stop_event),
                daemon=True
            )
            collector_threads.append(thread)
            thread.start()
        
        # 동기화된 프레임 처리
        active_videos = set(self.video_paths)
        latest_detections = {}
        latest_violations = {}
        frame_count = 0
        max_frames = 3000
        
        while active_videos and frame_count < max_frames:
            # 모든 비디오에서 프레임이 준비될 때까지 대기
            all_frames_ready = True
            current_frames = {}
            
            for video_path in active_videos.copy():
                try:
                    frame_data = frame_buffers[video_path].get(timeout=0.01)
                    
                    if frame_data['frame'] is None:
                        active_videos.discard(video_path)
                        if video_path in latest_detections:
                            del latest_detections[video_path]
                        if video_path in latest_violations:
                            del latest_violations[video_path]
                        continue
                    
                    current_frames[video_path] = frame_data
                    
                except queue.Empty:
                    all_frames_ready = False
                    break
            
            if not all_frames_ready:
                time.sleep(0.01)
                continue
            
            # 모든 프레임이 준비되었으면 동시에 처리
            frame_count += 1
            print(f"[SYNC] Processing frame {frame_count} for all videos")
            
            # 모든 카메라의 감지 결과 수집
            all_detections = []
            all_violations = []
            
            for video_path, frame_data in current_frames.items():
                detections = frame_data['detections']
                violations = frame_data['violations']
                
                if detections:
                    latest_detections[video_path] = detections
                    all_detections.extend(detections)
                
                if violations:
                    latest_violations[video_path] = violations
                    all_violations.extend(violations)
            
            # 모든 스레드에게 다음 프레임 처리 허가 신호 전송
            sync_event.set()
        
        # 스레드 정리
        stop_event.set()
        for thread in collector_threads:
            thread.join(timeout=1.0)
        
        return all_detections, all_violations
    
    def save_tracking_results(self, output_file="tracking_results.json"):
        """추적 결과 저장"""
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "videos": self.video_paths,
            "model": self.model_path,
            "ppe_model": self.ppe_detector.model_path,
            "total_detections": len(self.tracking_results),
            "results": self.tracking_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Tracking results saved to: {output_file}")
    
    def print_performance_summary(self):
        """성능 요약 출력"""
        print("\n" + "="*60)
        print("📊 INTEGRATED TRACKING PERFORMANCE SUMMARY")
        print("="*60)
        
        for camera_id, thread_logger in self.thread_performance_loggers.items():
            print(f"\n📊 Camera {camera_id} Performance Summary:")
            thread_logger.print_summary()
        
        print("="*60)

