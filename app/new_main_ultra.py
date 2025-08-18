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

from config import settings
from integrated_tracking_system_ultra import IntegratedTrackingSystemUltra


def main():
    """Ultralytics Tracking을 사용하는 메인 함수"""
    parser = argparse.ArgumentParser(
        description="Integrated ReID, Homography, and PPE Tracking System (Ultralytics)"
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
    parser.add_argument(
        '--backend_url',
        type=str,
        default="http://localhost:5000",
        help='Backend server URL.'
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
    
    # Ultralytics 기반 통합 추적 시스템 생성
    tracking_system = IntegratedTrackingSystemUltra(
        video_paths=args.videos,
        model_path=args.yolo_model,
        tracker_config=tracker_config,
        redis_conf=redis_conf,
        reid_conf=reid_config,
        calibration_files=calibration_files,
        backend_url=args.backend_url,
        ppe_model_path=args.ppe_model
    )
    
    print(f"▶ Processing {len(args.videos)} videos with Ultralytics tracking and PPE detection")
    for i, video_path in enumerate(args.videos):
        print(f"  Camera {i}: {video_path}")
        if i in calibration_files:
            print(f"    Calibration: {calibration_files[i]}")
        else:
            print(f"    Homography: Loaded from settings.py")
    
    # 멀티 비디오 추적 실행
    all_detections, all_violations = tracking_system.run_multi_video_tracking()
    print(f"▶ Total detections: {len(all_detections)}")
    print(f"▶ Total PPE violations: {len(all_violations)}")
    
    # 성능 요약 출력
    tracking_system.print_performance_summary()
    
    # 결과 저장
    tracking_system.save_tracking_results()


if __name__ == '__main__':
    main()
