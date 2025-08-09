# config/settings.py

# YOLO 모델 경로
YOLO_MODEL_PATH = "models/weights/bestcctv.pt"

# BYTETracker 설정 (원본과 동일)
TRACKER_CONFIG = {
    "track_thresh": 0.5,
    "match_thresh": 0.8,
    "track_buffer": 300,
    "mot20": False,
    "frame_rate": 30,
    "target_width": 640,  # 프레임 리사이즈 목표 너비
}

# Redis 설정
REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "camera_id": "cam01"
}

# ReID 설정 (원본의 복잡한 설정 반영)
REID_CONFIG = {
    "threshold": 0.5,  # 원본과 동일한 임계값
    "ttl": 300,  # seconds
    "frame_rate": 30,
    "feature_ttl": 3000,  # 100초 (원본과 동일)
    "similarity_threshold": 0.3,  # 더 관대한 매칭을 위해 낮춤
}

# 입력 비디오 경로 (원본과 동일)
VIDEO_INPUT_PATHS = [
    "test_video/KSEB02.mp4",
    "test_video/KSEB03.mp4"
]

# 좌표 변환 매트릭스 (원본과 동일)
HOMOGRAPHY_MATRIX = [
    [0.000030, -0.000119, 0.043679],
    [-0.000115, -0.000221, 0.290054],
    [-0.000199, -0.000943, 1.000000]
]

# 탐지 설정
DETECTION_CONFIG = {
    "person_classes": ["person", "saram"],  # 탐지할 클래스들
    "confidence_threshold": 0.5,
    "nms_threshold": 0.4,
}

# 멀티스레딩 설정 (원본과 동일)
MULTITHREADING_CONFIG = {
    "enabled": True,
    "max_frames": 300,  # 최대 처리 프레임 수 (약 10초)
    "frame_timeout": 0.1,  # 프레임 큐 타임아웃
    "thread_timeout": 1.0,  # 스레드 종료 타임아웃
}

# GUI 설정 (원본과 동일)
GUI_CONFIG = {
    "window_width": 800,
    "window_height": 600,
    "window_normal": True,  # 창 크기 조절 가능
    "display_fps": True,
    "show_coordinates": True,
    "show_global_id": True,
}

# Feature Extractor 설정 (원본과 동일)
FEATURE_EXTRACTOR_CONFIG = {
    "model_name": "osnet_ibn_x1_0",
    "model_path": None,  # 자동 다운로드
    "device": "auto",  # cuda if available, else cpu
    "target_size": (128, 256),  # feature extraction용 이미지 크기
    "padding_color": (0, 0, 0),  # 패딩 색상
}

# 카메라별 설정 (원본과 동일)
CAMERA_CONFIGS = {
    "camera_0": {
        "id": 0,
        "name": "KSEB02",
        "homography_matrix": HOMOGRAPHY_MATRIX,
        "position": "entrance"
    },
    "camera_1": {
        "id": 1,
        "name": "KSEB03", 
        "homography_matrix": HOMOGRAPHY_MATRIX,
        "position": "exit"
    }
}

# 로깅 설정 (원본과 동일)
LOGGING_CONFIG = {
    "level": "INFO",
    "show_debug": True,
    "show_coordinates": True,
    "show_reid_matches": True,
    "show_disappearing_objects": True,
}

# 성능 설정 (원본과 동일)
PERFORMANCE_CONFIG = {
    "use_gpu": True,
    "half_precision": True,  # GPU 사용시 half precision
    "batch_size": 1,
    "num_workers": 0,  # 멀티스레딩에서는 0으로 설정
}
