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
import json
import requests
from result.performance_logger import PerformanceLogger
from datetime import datetime

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


class HomographyManager:
    """호모그래피 매트릭스 관리 클래스"""
    
    def __init__(self):
        self.homography_matrices = {}
        self.calibration_data = {}
        # Unity 맵 좌표계 설정 (4개 좌표 방식)
        self.unity_map_corners = {
            0: [  # 카메라 0 (복도) - 왼쪽 상단부터 시계방향
                {"x": 3700, "y": -4088},      # 왼쪽 상단
                {"x": 3700, "y": -1700},    # 오른쪽 상단
                {"x": 10700, "y": 1080},   # 오른쪽 하단
                {"x": 10700, "y": -8700}      # 왼쪽 하단
            ],
            1: [  # 카메라 1 (방) - 왼쪽 상단부터 시계방향
                {"x": 5439, "y": -770},    # 왼쪽 상단
                {"x": 4000, "y": -1350},    # 오른쪽 상단
                {"x": 3220, "y": 408},  # 오른쪽 하단
                {"x": 4606, "y": 903}   # 왼쪽 하단
            ]
        }
    
    def set_unity_map_corners(self, camera_id, corners):
        """Unity 맵에서 각 카메라 영역의 4개 모서리 좌표 설정
        corners: [왼쪽상단, 오른쪽상단, 오른쪽하단, 왼쪽하단] 순서
        """
        if len(corners) != 4:
            print(f"Error: Need exactly 4 corners for camera {camera_id}")
            return False
        
        self.unity_map_corners[camera_id] = corners
        print(f"Camera {camera_id} Unity corners set:")
        for i, corner in enumerate(corners):
            print(f"  Corner {i+1}: ({corner['x']}, {corner['y']})")
        return True
    
    def set_unity_map_corners_from_coords(self, camera_id, x1, y1, x2, y2, x3, y3, x4, y4):
        """Unity 맵에서 각 카메라 영역의 4개 모서리 좌표 설정 (개별 좌표 입력)"""
        corners = [
            {"x": x1, "y": y1},  # 왼쪽 상단
            {"x": x2, "y": y2},  # 오른쪽 상단
            {"x": x3, "y": y3},  # 오른쪽 하단
            {"x": x4, "y": y4}   # 왼쪽 하단
        ]
        return self.set_unity_map_corners(camera_id, corners)
    
    def add_camera_calibration(self, camera_id, calibration_file):
        """카메라별 캘리브레이션 파일 로드"""
        try:
            with open(calibration_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.homography_matrices[camera_id] = np.array(data['homography_matrix'])
            self.calibration_data[camera_id] = data
            
            print(f"Camera {camera_id} calibration loaded: {calibration_file}")
            return True
            
        except Exception as e:
            print(f"Failed to load calibration for camera {camera_id}: {e}")
            return False
    
    def transform_coordinates(self, camera_id, x, y):
        """좌표 변환 (호모그래피 + Unity 맵 4개 좌표 변환)"""
        if camera_id not in self.homography_matrices:
            print(f"Warning: No homography matrix for camera {camera_id}")
            return x, y
        
        try:
            # 1단계: 호모그래피 변환 (카메라 좌표 → 실제 지면 좌표)
            point = np.array([[x, y]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(point, self.homography_matrices[camera_id])
            real_x, real_y = transformed[0][0], transformed[0][1]
            
            # 2단계: Unity 맵 4개 좌표로 변환
            if camera_id in self.unity_map_corners:
                unity_corners = self.unity_map_corners[camera_id]
                
                # 실제 지면 좌표를 0~1 범위로 정규화 (캘리브레이션 데이터 기준)
                if camera_id == 0:  # 카메라 0 (final01)
                    # 7.5m × 7.8m 영역을 0~1로 정규화
                    norm_x = max(0, min(1, real_x / 7.5))
                    norm_y = max(0, min(1, real_y / 7.8))
                elif camera_id == 1:  # 카메라 1 (final02)
                    # 9.0m × 8.2m 영역을 0~1로 정규화
                    norm_x = max(0, min(1, real_x / 9.0))
                    norm_y = max(0, min(1, real_y / 8.2))
                else:
                    # 기본값 (0~1 범위로 가정)
                    norm_x = max(0, min(1, real_x))
                    norm_y = max(0, min(1, real_y))
                
                # 정규화된 좌표를 Unity 맵의 4개 모서리 좌표로 변환
                # 4개 모서리 좌표를 사용한 바이리니어 보간
                unity_x = (unity_corners[0]["x"] * (1 - norm_x) * (1 - norm_y) +  # 왼쪽 상단
                          unity_corners[1]["x"] * norm_x * (1 - norm_y) +         # 오른쪽 상단
                          unity_corners[2]["x"] * norm_x * norm_y +               # 오른쪽 하단
                          unity_corners[3]["x"] * (1 - norm_x) * norm_y)          # 왼쪽 하단
                
                unity_y = (unity_corners[0]["y"] * (1 - norm_x) * (1 - norm_y) +  # 왼쪽 상단
                          unity_corners[1]["y"] * norm_x * (1 - norm_y) +         # 오른쪽 상단
                          unity_corners[2]["y"] * norm_x * norm_y +               # 오른쪽 하단
                          unity_corners[3]["y"] * (1 - norm_x) * norm_y)          # 왼쪽 하단
                
                print(f"[UNITY] Camera {camera_id}: Image({x:.1f}, {y:.1f}) -> Real({real_x:.3f}, {real_y:.3f}) -> Norm({norm_x:.3f}, {norm_y:.3f}) -> Unity({unity_x:.1f}, {unity_y:.1f})")
                return unity_x, unity_y
            else:
                print(f"Warning: No Unity corners for camera {camera_id}")
                return real_x, real_y
                
        except Exception as e:
            print(f"Coordinate transformation failed: {e}")
            return x, y
    
    def get_calibration_info(self, camera_id):
        """캘리브레이션 정보 반환"""
        if camera_id in self.calibration_data:
            return self.calibration_data[camera_id]
        return None
    
    def get_unity_corners(self, camera_id):
        """Unity 맵 4개 모서리 좌표 반환"""
        return self.unity_map_corners.get(camera_id, None)
    
    def print_unity_corners(self, camera_id):
        """Unity 맵 4개 모서리 좌표 출력"""
        corners = self.get_unity_corners(camera_id)
        if corners:
            print(f"Camera {camera_id} Unity corners:")
            for i, corner in enumerate(corners):
                print(f"  Corner {i+1}: ({corner['x']}, {corner['y']})")
        else:
            print(f"No Unity corners set for camera {camera_id}")


class BackendClient:
    """백엔드 서버와 통신하는 클라이언트"""
    
    def __init__(self, backend_url="http://localhost:5000"):
        self.backend_url = backend_url
        self.workers_endpoint = f"{backend_url}/workers"
    
    def send_worker_data(self, worker_data):
        """워커 데이터를 백엔드로 전송"""
        try:
            response = requests.post(
                self.workers_endpoint,
                json=worker_data,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            
            if response.status_code == 200:
                print(f"[BACKEND] Worker data sent successfully: {worker_data}")
                return True
            else:
                print(f"[BACKEND] Failed to send worker data. Status: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"[BACKEND] Error sending worker data: {e}")
            return False


class IntegratedTrackingSystem:
    """
    ReID 기능과 호모그래피 기능을 통합한 객체 추적 시스템
    두 개의 영상에서 객체 추적이 잘 되도록 호모그래피 좌표를 직접 입력하는 방식
    """
    
    def __init__(self, video_paths=None, model_path=None, tracker_config=None, 
                 redis_conf=None, reid_conf=None, calibration_files=None, backend_url=None):
        # 설정 로드
        self.model_path = model_path or settings.YOLO_MODEL_PATH
        self.tracker_config = tracker_config or settings.TRACKER_CONFIG
        redis_conf = redis_conf or settings.REDIS_CONFIG
        reid_conf = reid_conf or settings.REID_CONFIG
        
        # 비디오 경로 설정
        self.video_paths = video_paths or settings.VIDEO_INPUT_PATHS
        
        # 호모그래피 매니저 초기화
        self.homography_manager = HomographyManager()
        
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
        
        print("Integrated Tracking System initialized")
        print(f"Videos: {self.video_paths}")
        print(f"Model: {self.model_path}")
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
            
            # 호모그래피 변환
            real_x, real_y = self.homography_manager.transform_coordinates(camera_id, point_x, point_y)
            
            # 결과 저장
            detection_data = {
                "camera_id": int(camera_id),
                "worker_id": int(global_id),
                "x": float(real_x),
                "y": float(real_y),
                "frame_id": int(frame_id),
                "local_id": int(local_id),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "image_coords": [float(point_x), float(point_y)]
            }
            frame_detections.append(detection_data)
            
            # 바운딩 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID:{global_id}', (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 좌표 정보 표시
            coord_text = f"({real_x:.1f}, {real_y:.1f})"
            cv2.putText(frame, coord_text, (x1, y2 + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # 성능 데이터 로깅
        logger.log_frame_performance()
        
        return frame, frame_detections
    
    def send_detections_to_backend(self, detections):
        """감지 결과를 백엔드로 전송"""
        if not detections:
            return
        
        # 백엔드 형식에 맞게 데이터 변환
        worker_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "workers": []
        }
        
        for det in detections:
            worker_info = {
                "worker_id": det.get("worker_id") or det.get("workerID"),
                "camera_id": det.get("camera_id") or det.get("cameraID"),
                "x": det.get("x") if det.get("x") is not None else det.get("position_X"),
                "y": det.get("y") if det.get("y") is not None else det.get("position_Y"),
                "frame_id": det.get("frame_id"),
                "local_id": det.get("local_id"),
            }
            worker_data["workers"].append(worker_info)
        
        # 백엔드로 전송
        self.backend_client.send_worker_data(worker_data)
    
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
            processed_frame, detections = self.process_frame_with_reid_and_homography(
                frame, processed_frame_id, camera_id, detector
            )
            
            # 백엔드로 전송
            self.send_detections_to_backend(detections)
            
            # 결과 저장
            frame_data = {
                'video_path': video_path,
                'frame': processed_frame,
                'detections': detections,
                'frame_id': processed_frame_id
            }
            
            # 버퍼에 저장
            frame_buffer.put(frame_data)
            
            # 동기화 대기
            sync_event.wait(0.1)
            sync_event.clear()
        
        cap.release()
        frame_buffer.put({'video_path': video_path, 'frame': None, 'detections': None, 'frame_id': -1})
    
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
        
        # GUI 처리
        active_videos = set(self.video_paths)
        latest_detections = {}
        
        # 각 비디오별 창 생성
        window_names = {}
        for i, video_path in enumerate(self.video_paths):
            window_name = f"Camera {i} - {Path(video_path).name}"
            window_names[video_path] = window_name
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
        
        # 동기화된 프레임 처리
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
            
            for video_path, frame_data in current_frames.items():
                frame = frame_data['frame']
                detections = frame_data['detections']
                frame_id = frame_data['frame_id']
                
                if detections:
                    latest_detections[video_path] = detections
                    all_detections.extend(detections)
                
                # 각 비디오를 별도의 창에 표시
                if frame is not None:
                    window_name = window_names[video_path]
                    cv2.imshow(window_name, frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # 모든 스레드에게 다음 프레임 처리 허가 신호 전송
            sync_event.set()
        
        # 스레드 정리
        stop_event.set()
        for thread in collector_threads:
            thread.join(timeout=1.0)
        
        # 모든 창 닫기
        for window_name in window_names.values():
            cv2.destroyWindow(window_name)
        
        return all_detections
    
    def save_tracking_results(self, output_file="tracking_results.json"):
        """추적 결과 저장"""
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "videos": self.video_paths,
            "model": self.model_path,
            "total_detections": len(self.tracking_results),
            "results": self.tracking_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Tracking results saved to: {output_file}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Integrated ReID and Homography Tracking System"
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
        '--calibration_files',
        nargs='+',
        type=str,
        help='List of calibration files for each camera (in order).'
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
        '--backend_url',
        type=str,
        default="http://localhost:5000",
        help='Backend server URL.'
    )
    parser.add_argument(
        '--unity_corners',
        nargs='+',
        type=str,
        help='Unity map corners for each camera in format "camera_id:x1,y1,x2,y2,x3,y3,x4,y4" (clockwise from top-left)'
    )
    args = parser.parse_args()
    
    # 캘리브레이션 파일 매핑
    calibration_files = {}
    if args.calibration_files:
        for i, calib_file in enumerate(args.calibration_files):
            calibration_files[i] = calib_file
    else:
        # 캘리브레이션 파일이 지정되지 않으면 settings에서 자동 로드
        print("No calibration files specified. Using homography matrices from settings.py")
    
    # 시스템 초기화
    tracker_config = settings.TRACKER_CONFIG
    reid_config = settings.REID_CONFIG
    
    redis_conf = {
        "host": args.redis_host,
        "port": args.redis_port,
        "camera_id": "camera_0"
    }
    
    # 통합 추적 시스템 생성
    tracking_system = IntegratedTrackingSystem(
        video_paths=args.videos,
        model_path=args.yolo_model,
        tracker_config=tracker_config,
        redis_conf=redis_conf,
        reid_conf=reid_config,
        calibration_files=calibration_files,
        backend_url=args.backend_url
    )
    
    # Unity 좌표 설정
    if args.unity_corners:
        for corner_str in args.unity_corners:
            try:
                # 형식: "camera_id:x1,y1,x2,y2,x3,y3,x4,y4"
                camera_part, coords_part = corner_str.split(':', 1)
                camera_id = int(camera_part)
                coords = [float(x) for x in coords_part.split(',')]
                
                if len(coords) == 8:
                    tracking_system.homography_manager.set_unity_map_corners_from_coords(
                        camera_id, coords[0], coords[1], coords[2], coords[3], 
                        coords[4], coords[5], coords[6], coords[7]
                    )
                else:
                    print(f"Error: Need exactly 8 coordinates for camera {camera_id}")
            except Exception as e:
                print(f"Error parsing Unity corners: {corner_str}, Error: {e}")
    
    # Unity 좌표 설정 (예시 - 실제 값으로 수정 필요)
    # 복도와 방의 Unity 좌표를 여기서 설정하거나 명령행 인자로 전달
    # tracking_system.homography_manager.set_unity_map_corners_from_coords(
    #     0,  # 카메라 0 (복도)
    #     0, 0,    # 왼쪽 상단
    #     100, 0,  # 오른쪽 상단
    #     100, 50, # 오른쪽 하단
    #     0, 50    # 왼쪽 하단
    # )
    # tracking_system.homography_manager.set_unity_map_corners_from_coords(
    #     1,  # 카메라 1 (방)
    #     100, 0,   # 왼쪽 상단
    #     200, 0,   # 오른쪽 상단
    #     200, 100, # 오른쪽 하단
    #     100, 100  # 왼쪽 하단
    # )
    
    print(f"▶ Processing {len(args.videos)} videos with integrated tracking")
    for i, video_path in enumerate(args.videos):
        print(f"  Camera {i}: {video_path}")
        if i in calibration_files:
            print(f"    Calibration: {calibration_files[i]}")
        else:
            print(f"    Homography: Loaded from settings.py")
        tracking_system.homography_manager.print_unity_corners(i)
    
    # 멀티 비디오 추적 실행
    all_detections = tracking_system.run_multi_video_tracking()
    print(f"▶ Total detections: {len(all_detections)}")
    
    # 성능 요약 출력
    print("\n" + "="*60)
    print("📊 INTEGRATED TRACKING PERFORMANCE SUMMARY")
    print("="*60)
    
    for camera_id, thread_logger in tracking_system.thread_performance_loggers.items():
        print(f"\n📊 Camera {camera_id} Performance Summary:")
        thread_logger.print_summary()
    
    print("="*60)
    
    # 결과 저장
    tracking_system.save_tracking_results()


if __name__ == '__main__':
    main()
