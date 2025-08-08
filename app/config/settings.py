# config/settings.py

# YOLO 모델 경로
YOLO_MODEL_PATH = "yolov8n.pt"

# BYTETracker 설정
TRACKER_CONFIG = {
    "track_thresh": 0.5,
    "track_buffer": 30,
    "match_thresh": 0.8,
}

# Redis 설정
REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "camera_id": "cam01"
}

# ReID 설정
REID_CONFIG = {
    "threshold": 0.7,
    "ttl": 300  # seconds
}

# 입력 비디오 경로
VIDEO_INPUT_PATH = "sample.mp4"
