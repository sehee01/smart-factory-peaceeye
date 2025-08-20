import argparse
import sys
from pathlib import Path

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRA_PATHS = [
    str(PROJECT_ROOT),                      # 루트 자체 (app, frontend 등 import 가능)
    str(PROJECT_ROOT / "deep-person-reid-master"),  # 필요한 경우만
    str(PROJECT_ROOT / "app" / "models" / "mapping"),   # point_transformer 경로 수정
]

for p in EXTRA_PATHS:
    if p not in sys.path:
        sys.path.insert(0, p)

# np.float 호환성 문제 해결
import numpy as np
if not hasattr(np, 'float'):
    np.float = float

import cv2
import time
import threading
from config import settings
from integrated_tracking_system_ultra import IntegratedTrackingSystemUltra


class GUIVideoProcessor:
    """GUI 화면을 표시하는 비디오 처리 클래스 (ByteTrack + Pre-Registration Matcher만 사용)"""
    
    def __init__(self, tracking_system):
        self.tracking_system = tracking_system
        self.video_paths = tracking_system.video_paths
        self.stop_event = threading.Event()
        self.detectors = {}
        
        # GUI 설정
        self.window_names = []
        self.video_windows = {}
        
        # 성능 측정
        self.fps_counters = {}
        self.frame_times = {}
        
    def create_detector_for_thread(self, camera_id):
        """스레드별 독립적인 detector 생성"""
        if camera_id not in self.detectors:
            self.detectors[camera_id] = self.tracking_system.create_detector_for_thread()
        return self.detectors[camera_id]
    
    def process_video_thread(self, video_path, camera_id):
        """비디오 처리 스레드 (GUI 표시용) - ByteTrack + Pre-Registration Matcher만 사용"""
        try:
            detector = self.create_detector_for_thread(camera_id)
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Could not open video: {video_path}")
                return
            
            # 윈도우 생성
            window_name = f"Camera {camera_id} - {Path(video_path).name} (ByteTrack + Pre-Reg)"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1920, 1080)
            self.window_names.append(window_name)
            
            frame_id = 0
            start_time = time.time()
            
            print(f"🎥 Started processing Camera {camera_id}: {video_path} (ByteTrack + Pre-Registration)")
            
            while cap.isOpened() and not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print(f"📹 Camera {camera_id} video ended")
                    break
                
                frame_id += 1
                
                # 프레임 스킵 (성능 향상)
                if frame_id % 2 != 0:  # 2프레임마다 처리
                    continue
                
                # 처리 시간 측정
                frame_start_time = time.time()
                
                # ByteTrack + Pre-Registration Matcher만 사용
                track_list = detector.detect_and_track(frame, frame_id)
                
                # 결과를 기존 형식에 맞게 변환
                detections = []
                violations = []  # PPE 검사는 일단 비활성화
                
                for track in track_list:
                    track_id = track["track_id"]
                    bbox = track["bbox"]
                    confidence = track.get("confidence", 0.0)
                    
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Pre-Registration Matcher만 사용하여 Global ID 결정
                    global_id = self.match_with_pre_registration_only(frame, bbox, camera_id, frame_id, track_id)
                    
                    # 바운딩 박스 그리기
                    color = (0, 255, 0) if global_id == track_id else (0, 255, 255)  # 매칭되면 노란색
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'ID:{global_id} ({confidence:.2f})', 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # detection 데이터 생성
                    detection_data = {
                        "cameraID": camera_id,
                        "workerID": global_id,  # Pre-Registration 결과 사용
                        "position_X": (x1 + x2) / 2,  # 간단한 중심점 계산
                        "position_Y": (y1 + y2) / 2,
                        "frame_id": frame_id
                    }
                    detections.append(detection_data)
                
                processed_frame = frame  # 원본 프레임 사용
                
                # FPS 계산
                frame_time = time.time() - frame_start_time
                if camera_id not in self.fps_counters:
                    self.fps_counters[camera_id] = []
                self.fps_counters[camera_id].append(frame_time)
                
                # 최근 30프레임의 평균 FPS 계산
                if len(self.fps_counters[camera_id]) > 30:
                    self.fps_counters[camera_id] = self.fps_counters[camera_id][-30:]
                
                avg_fps = 1.0 / (sum(self.fps_counters[camera_id]) / len(self.fps_counters[camera_id])) if self.fps_counters[camera_id] else 0
                
                # GUI 정보 추가
                self.add_gui_info(processed_frame, camera_id, frame_id, len(detections), avg_fps, violations)
                
                # 캐시 통계 출력 (30프레임마다)
                if frame_id % 30 == 0:
                    self.tracking_system.matching_cache_manager.print_cache_stats()
                
                # 화면에 표시
                cv2.imshow(window_name, processed_frame)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.stop_event.set()
                    break
                elif key == ord('s'):  # 스크린샷 저장
                    screenshot_path = f"screenshot_camera_{camera_id}_{frame_id}_bytetrack_pre_reg.jpg"
                    cv2.imwrite(screenshot_path, processed_frame)
                    print(f"📸 Screenshot saved: {screenshot_path}")
            
            cap.release()
            print(f"✅ Camera {camera_id} processing completed (ByteTrack + Pre-Registration)")
            
        except Exception as e:
            print(f"❌ Error in camera {camera_id} processing: {e}")
            import traceback
            traceback.print_exc()
    
    def match_with_pre_registration_only(self, frame, bbox, camera_id, frame_id, local_id):
        """
        Pre-Registration Matcher만 사용하여 Global ID 결정 (캐시 기반)
        객체가 처음 탐지될 때만 사전 등록 매칭을 수행하고, 이후에는 캐시된 결과 사용
        """
        # 1. 캐시 확인 (이미 매칭된 객체)
        cached_global_id = self.tracking_system.matching_cache_manager.get_cached_global_id(camera_id, local_id)
        if cached_global_id is not None:
            return cached_global_id
        
        # 2. 새로운 객체: 사전 등록 매칭 시도
        print(f"[DEBUG] 🆕 새로운 객체 탐지: Camera {camera_id}, Local ID {local_id}")
        
        try:
            # 이미지 크롭
            crop_img = self.tracking_system.image_processor.crop_bbox_from_frame(frame, bbox)
            if crop_img.size == 0:
                print(f"[DEBUG] 이미지 크롭 실패 - Local ID {local_id}를 Global ID로 사용")
                self.tracking_system.matching_cache_manager.add_matching_cache(camera_id, local_id, local_id)
                return local_id
            
            # Feature 추출
            feature = self.tracking_system.image_processor.extract_feature(crop_img)
            if feature is None:
                print(f"[DEBUG] Feature 추출 실패 - Local ID {local_id}를 Global ID로 사용")
                self.tracking_system.matching_cache_manager.add_matching_cache(camera_id, local_id, local_id)
                return local_id
            
            # Pre-Registration Matcher 사용
            pre_reg_match = self.tracking_system.reid.pre_reg_matcher.match(feature)
            
            if pre_reg_match:
                # 매칭 성공: 사전 등록된 Global ID 사용
                print(f"[DEBUG] ✅ Pre-Registration 매칭 성공: Local {local_id} -> Global {pre_reg_match}")
                self.tracking_system.matching_cache_manager.add_matching_cache(camera_id, local_id, pre_reg_match)
                return pre_reg_match
            else:
                # 매칭 실패: Local ID를 Global ID로 사용
                print(f"[DEBUG] ❌ Pre-Registration 매칭 실패: Local ID {local_id}를 Global ID로 사용")
                self.tracking_system.matching_cache_manager.add_matching_cache(camera_id, local_id, local_id)
                return local_id
                
        except Exception as e:
            print(f"[DEBUG] ❌ Pre-Registration 매칭 중 오류: {e} - Local ID {local_id}를 Global ID로 사용")
            self.tracking_system.matching_cache_manager.add_matching_cache(camera_id, local_id, local_id)
            return local_id
    
    def add_gui_info(self, frame, camera_id, frame_id, detection_count, fps, violations):
        """GUI에 정보 추가 (ByteTrack + Pre-Registration Matcher만 사용)"""
        # 배경 정보 패널
        cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 120), (255, 255, 255), 2)
        
        # 정보 텍스트 (ByteTrack + Pre-Registration Matcher만 사용)
        info_lines = [
            f"Camera: {camera_id}",
            f"Frame: {frame_id}",
            f"Tracks: {detection_count}",
            f"FPS: {fps:.1f}",
            f"Mode: ByteTrack + Pre-Reg"  # ByteTrack + Pre-Registration만 사용 표시
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = 35 + i * 18
            cv2.putText(frame, line, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # ByteTrack + Pre-Registration 모드 표시
        cv2.rectangle(frame, (10, 130), (400, 160), (0, 255, 255), -1)
        cv2.putText(frame, f"BYTETRACK + PRE-REGISTRATION", (20, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    def run_multi_video_gui(self):
        """멀티 비디오 GUI 실행"""
        print("🚀 Starting Multi-Video GUI Processing (ByteTrack + Pre-Registration)...")
        print(f"📹 Processing {len(self.video_paths)} videos")
        
        # 각 비디오별 스레드 생성
        threads = []
        for i, video_path in enumerate(self.video_paths):
            thread = threading.Thread(
                target=self.process_video_thread,
                args=(video_path, i),
                daemon=True
            )
            threads.append(thread)
            thread.start()
        
        try:
            # 메인 루프 (키 입력 처리)
            while not self.stop_event.is_set():
                # 키 입력 확인
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("🛑 Quit requested by user")
                    break
                
                # 모든 윈도우가 닫혔는지 확인 (안전하게)
                try:
                    if self.window_names and cv2.getWindowProperty(self.window_names[0], cv2.WND_PROP_VISIBLE) < 1:
                        print("🛑 Window closed by user")
                        break
                except:
                    pass  # 윈도우가 아직 생성되지 않았을 수 있음
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        finally:
            # 정리
            self.stop_event.set()
            for thread in threads:
                thread.join(timeout=1.0)
            
            cv2.destroyAllWindows()
            print("✅ GUI processing completed (ByteTrack + Pre-Registration)")


def main():
    """GUI 버전 메인 함수 (ByteTrack + Pre-Registration Matcher만 사용)"""
    parser = argparse.ArgumentParser(
        description="ByteTrack + Pre-Registration Matcher Only Tracking System with GUI (Ultralytics)"
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
        default=settings.YOLO_MODEL_PATH,
        help='Path to the YOLOv8 model file for person detection.'
    )
    parser.add_argument(
        '--ppe_model',
        type=str,
        default=settings.PPE_MODEL_PATH,
        help='Path to the PPE detection model file.'
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
    args = parser.parse_args()
    
    # 캘리브레이션 파일 매핑
    calibration_files = {}
    if args.calibration_files:
        for i, calib_file in enumerate(args.calibration_files):
            calibration_files[i] = calib_file
    else:
        print("No calibration files specified. Using homography matrices from settings.py")
    
    # 시스템 초기화 (ByteTrack + Pre-Registration Matcher만 사용)
    tracker_config = settings.TRACKER_CONFIG
    reid_config = settings.REID_CONFIG  # 설정은 로드하지만 일부만 사용
    
    redis_conf = {
        "host": args.redis_host,
        "port": args.redis_port,
        "camera_id": "camera_0"
    }
    
    # Ultralytics 기반 통합 추적 시스템 생성
    tracking_system = IntegratedTrackingSystemUltra(
        video_paths=args.videos,
        model_path=args.yolo_model,
        tracker_config=tracker_config,
        redis_conf=redis_conf,
        reid_conf=reid_config,
        calibration_files=calibration_files,
        backend_url=None,  # 백엔드 전송 비활성화
        ppe_model_path=args.ppe_model
    )
    
    print(f"▶ Processing {len(args.videos)} videos with GUI display (ByteTrack + Pre-Registration)")
    print(f"⚠️  Only ByteTrack + Pre-Registration Matcher will be used")
    for i, video_path in enumerate(args.videos):
        print(f"  Camera {i}: {video_path}")
        if i in calibration_files:
            print(f"    Calibration: {calibration_files[i]}")
        else:
            print(f"    Homography: Loaded from settings.py")
    
    # GUI 비디오 프로세서 생성 및 실행
    gui_processor = GUIVideoProcessor(tracking_system)
    gui_processor.run_multi_video_gui()
    
    print("🎉 GUI processing completed (ByteTrack + Pre-Registration)!")


if __name__ == '__main__':
    main()
