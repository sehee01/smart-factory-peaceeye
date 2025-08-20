import argparse
from pathlib import Path
import time
import threading
import numpy as np
import cv2

# np.float 호환성 (일부 OpenCV/의존 라이브러리 대비)
if not hasattr(np, "float"):
    np.float = float

# 내부 모듈 (새 구조)
from app.config import settings
from app.core.tracking_system_ultra import IntegratedTrackingSystemUltra


class GUIVideoProcessor:
    """GUI 화면을 표시하는 비디오 처리 클래스 (ByteTrack + Pre-Registration Matcher만 사용)"""

    def __init__(self, tracking_system: IntegratedTrackingSystemUltra):
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

    def create_detector_for_thread(self, camera_id: int):
        """스레드별 독립적인 detector 생성"""
        if camera_id not in self.detectors:
            self.detectors[camera_id] = self.tracking_system.create_detector_for_thread()
        return self.detectors[camera_id]

    def process_video_thread(self, video_path: str, camera_id: int):
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
            print(f"🎥 Started processing Camera {camera_id}: {video_path} (ByteTrack + Pre-Registration)")

            while cap.isOpened() and not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print(f"📹 Camera {camera_id} video ended")
                    break

                frame_id += 1

                # 프레임 스킵 (2프레임마다 처리)
                if frame_id % 2 != 0:
                    continue

                # 처리 시간 측정 시작
                frame_start_time = time.time()

                # ByteTrack + Pre-Registration Matcher만 사용
                track_list = detector.detect_and_track(frame, frame_id)

                # 결과 변환
                detections = []
                violations = []  # PPE 검사는 GUI 경량모드에서는 비활성화

                for track in track_list:
                    track_id = track["track_id"]
                    bbox = track["bbox"]
                    confidence = track.get("confidence", 0.0)

                    x1, y1, x2, y2 = map(int, bbox)

                    # Pre-Registration Matcher만 사용하여 Global ID 결정 (캐시 우선)
                    global_id = self.match_with_pre_registration_only(
                        frame, bbox, camera_id, frame_id, track_id
                    )

                    # 바운딩 박스 그리기
                    color = (0, 255, 0) if global_id == track_id else (0, 255, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"ID:{global_id} ({confidence:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                    # detection 데이터 (간단 중심점)
                    detection_data = {
                        "cameraID": camera_id,
                        "workerID": global_id,
                        "position_X": (x1 + x2) / 2.0,
                        "position_Y": (y1 + y2) / 2.0,
                        "frame_id": frame_id,
                    }
                    detections.append(detection_data)

                processed_frame = frame  # 원본 프레임 사용

                # FPS 계산
                frame_time = time.time() - frame_start_time
                if camera_id not in self.fps_counters:
                    self.fps_counters[camera_id] = []
                self.fps_counters[camera_id].append(frame_time)

                if len(self.fps_counters[camera_id]) > 30:
                    self.fps_counters[camera_id] = self.fps_counters[camera_id][-30:]

                avg_fps = (
                    1.0
                    / (sum(self.fps_counters[camera_id]) / len(self.fps_counters[camera_id]))
                    if self.fps_counters[camera_id]
                    else 0.0
                )

                # GUI 정보 추가
                self.add_gui_info(processed_frame, camera_id, frame_id, len(detections), avg_fps, violations)

                # 캐시 통계 출력 (30프레임마다)
                if frame_id % 30 == 0:
                    self.tracking_system.matching_cache_manager.print_cache_stats()

                # 화면 표시
                cv2.imshow(window_name, processed_frame)

                # 키 입력
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.stop_event.set()
                    break
                elif key == ord("s"):  # 스크린샷 저장
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
        # 1) 캐시 확인
        cached_global_id = self.tracking_system.matching_cache_manager.get_cached_global_id(camera_id, local_id)
        if cached_global_id is not None:
            return cached_global_id

        # 2) 신규 객체: 사전 등록 매칭 시도
        print(f"[DEBUG] 🆕 새로운 객체 탐지: Camera {camera_id}, Local ID {local_id}")

        try:
            crop_img = self.tracking_system.image_processor.crop_bbox_from_frame(frame, bbox)
            if crop_img.size == 0:
                print(f"[DEBUG] 이미지 크롭 실패 - Local ID {local_id}를 Global ID로 사용")
                self.tracking_system.matching_cache_manager.add_matching_cache(camera_id, local_id, local_id)
                return local_id

            feature = self.tracking_system.image_processor.extract_feature(crop_img)
            if feature is None:
                print(f"[DEBUG] Feature 추출 실패 - Local ID {local_id}를 Global ID로 사용")
                self.tracking_system.matching_cache_manager.add_matching_cache(camera_id, local_id, local_id)
                return local_id

            pre_reg_match = self.tracking_system.reid.pre_reg_matcher.match(feature)

            if pre_reg_match:
                print(f"[DEBUG] ✅ Pre-Registration 매칭 성공: Local {local_id} -> Global {pre_reg_match}")
                self.tracking_system.matching_cache_manager.add_matching_cache(camera_id, local_id, pre_reg_match)
                return pre_reg_match
            else:
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

        info_lines = [
            f"Camera: {camera_id}",
            f"Frame: {frame_id}",
            f"Tracks: {detection_count}",
            f"FPS: {fps:.1f}",
            "Mode: ByteTrack + Pre-Reg",
        ]

        for i, line in enumerate(info_lines):
            y_pos = 35 + i * 18
            cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 모드 배지
        cv2.rectangle(frame, (10, 130), (400, 160), (0, 255, 255), -1)
        cv2.putText(frame, "BYTETRACK + PRE-REGISTRATION", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    def run_multi_video_gui(self):
        """멀티 비디오 GUI 실행"""
        print("🚀 Starting Multi-Video GUI Processing (ByteTrack + Pre-Registration)...")
        print(f"📹 Processing {len(self.video_paths)} videos")

        # 각 비디오별 스레드 생성
        threads = []
        for i, video_path in enumerate(self.video_paths):
            thread = threading.Thread(target=self.process_video_thread, args=(video_path, i), daemon=True)
            threads.append(thread)
            thread.start()

        try:
            # 메인 루프 (키 입력 처리)
            while not self.stop_event.is_set():
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("🛑 Quit requested by user")
                    break

                # 모든 윈도우가 닫혔는지 확인
                try:
                    if self.window_names and cv2.getWindowProperty(self.window_names[0], cv2.WND_PROP_VISIBLE) < 1:
                        print("🛑 Window closed by user")
                        break
                except Exception:
                    pass

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
        "--videos", nargs="+", type=str, default=settings.VIDEO_INPUT_PATHS, help="List of video file paths."
    )
    parser.add_argument(
        "--yolo_model", type=str, default=settings.YOLO_MODEL_PATH, help="Path to the YOLOv8 model file for person detection."
    )
    parser.add_argument(
        "--ppe_model", type=str, default=settings.PPE_MODEL_PATH, help="Path to the PPE detection model file."
    )
    parser.add_argument(
        "--calibration_files", nargs="+", type=str, help="List of calibration files for each camera (in order)."
    )
    parser.add_argument("--redis_host", type=str, default="localhost", help="Redis server host.")
    parser.add_argument("--redis_port", type=int, default=6379, help="Redis server port.")
    args = parser.parse_args()

    # 캘리브레이션 파일 매핑 (현재 settings에서 HOMOGRAPHY 사용, 옵션)
    calibration_files = {}
    if args.calibration_files:
        for i, calib_file in enumerate(args.calibration_files):
            calibration_files[i] = calib_file
    else:
        print("No calibration files specified. Using homography matrices from settings.py")

    tracker_config = settings.TRACKER_CONFIG
    reid_config = settings.REID_CONFIG

    redis_conf = {"host": args.redis_host, "port": args.redis_port, "camera_id": "camera_0"}

    # 통합 추적 시스템 생성 (백엔드 전송 비활성화하려면 backend_url=None 유지)
    tracking_system = IntegratedTrackingSystemUltra(
        video_paths=args.videos,
        model_path=args.yolo_model,
        tracker_config=tracker_config,
        redis_conf=redis_conf,
        reid_conf=reid_config,
        calibration_files=calibration_files,
        backend_url=None,  # 백엔드 전송 비활성화
        ppe_model_path=args.ppe_model,
    )

    print(f"▶ Processing {len(args.videos)} videos with GUI display (ByteTrack + Pre-Registration)")
    print("⚠️  Only ByteTrack + Pre-Registration Matcher will be used")
    for i, video_path in enumerate(args.videos):
        print(f"  Camera {i}: {video_path}")
        if i in calibration_files:
            print(f"    Calibration: {calibration_files[i]}")
        else:
            print(f"    Homography: Loaded from settings.py")

    # GUI 비디오 프로세서 실행
    gui_processor = GUIVideoProcessor(tracking_system)
    gui_processor.run_multi_video_gui()

    print("🎉 GUI processing completed (ByteTrack + Pre-Registration)!")


if __name__ == "__main__":
    main()
