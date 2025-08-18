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
    """GUI 화면을 표시하는 비디오 처리 클래스"""
    
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
        """비디오 처리 스레드 (GUI 표시용)"""
        try:
            detector = self.create_detector_for_thread(camera_id)
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Could not open video: {video_path}")
                return
            
            # 윈도우 생성
            window_name = f"Camera {camera_id} - {Path(video_path).name}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1920, 1080)
            self.window_names.append(window_name)
            
            frame_id = 0
            start_time = time.time()
            
            print(f"🎥 Started processing Camera {camera_id}: {video_path}")
            
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
                
                # 프레임 처리 (ReID, 호모그래피, PPE 포함)
                processed_frame, detections, violations = self.tracking_system.process_frame_with_reid_and_homography(
                    frame, frame_id, camera_id, detector
                )
                
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
                
                # 화면에 표시
                cv2.imshow(window_name, processed_frame)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.stop_event.set()
                    break
                elif key == ord('s'):  # 스크린샷 저장
                    screenshot_path = f"screenshot_camera_{camera_id}_{frame_id}.jpg"
                    cv2.imwrite(screenshot_path, processed_frame)
                    print(f"📸 Screenshot saved: {screenshot_path}")
            
            cap.release()
            print(f"✅ Camera {camera_id} processing completed")
            
        except Exception as e:
            print(f"❌ Error in camera {camera_id} processing: {e}")
            import traceback
            traceback.print_exc()
    
    def add_gui_info(self, frame, camera_id, frame_id, detection_count, fps, violations):
        """GUI에 정보 추가"""
        # 배경 정보 패널
        cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 120), (255, 255, 255), 2)
        
        # 정보 텍스트
        info_lines = [
            f"Camera: {camera_id}",
            f"Frame: {frame_id}",
            f"Detections: {detection_count}",
            f"FPS: {fps:.1f}",
            f"PPE Violations: {len(violations)}"
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = 35 + i * 18
            cv2.putText(frame, line, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # PPE 위반 알림
        if violations:
            cv2.rectangle(frame, (10, 130), (400, 160), (0, 0, 255), -1)
            cv2.putText(frame, f"PPE VIOLATION DETECTED!", (20, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def run_multi_video_gui(self):
        """멀티 비디오 GUI 실행"""
        print("🚀 Starting Multi-Video GUI Processing...")
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
            print("✅ GUI processing completed")


def main():
    """GUI 버전 메인 함수"""
    parser = argparse.ArgumentParser(
        description="Integrated ReID, Homography, and PPE Tracking System with GUI (Ultralytics)"
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
        help='Path to the YOLOv8 model file for person detection.'
    )
    parser.add_argument(
        '--ppe_model',
        type=str,
        default="models/weights/best_yolo11n.pt",
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
    
    # 시스템 초기화
    tracker_config = settings.TRACKER_CONFIG
    reid_config = settings.REID_CONFIG
    
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
    
    print(f"▶ Processing {len(args.videos)} videos with GUI display")
    for i, video_path in enumerate(args.videos):
        print(f"  Camera {i}: {video_path}")
        if i in calibration_files:
            print(f"    Calibration: {calibration_files[i]}")
        else:
            print(f"    Homography: Loaded from settings.py")
    
    # GUI 비디오 프로세서 생성 및 실행
    gui_processor = GUIVideoProcessor(tracking_system)
    gui_processor.run_multi_video_gui()
    
    print("🎉 GUI processing completed!")


if __name__ == '__main__':
    main()
