# smart-factory-peaceeye/app/config/settings_ultra.py
# 바운딩 박스 안정화 기반 매칭을 위한 설정 파일

import os
from pathlib import Path

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 비디오 입력 경로
VIDEO_INPUT_PATHS = [
    "test_video/TEST100.mp4",
    # "test_video/TEST1.mp4"
]

# YOLO 모델 경로
YOLO_MODEL_PATH = "models/weights/yolo11x.engine"
PPE_MODEL_PATH = "models/weights/best_yolo11n.pt"

# ByteTrack 설정
TRACKER_CONFIG = {
    "track_high_thresh": 0.5,      # 높은 임계값
    "track_low_thresh": 0.1,       # 낮은 임계값
    "new_track_thresh": 0.6,       # 새로운 트랙 생성 임계값
    "track_buffer": 30,            # 트랙 버퍼 크기
    "match_thresh": 0.8,           # 매칭 임계값
    "min_box_area": 10,            # 최소 박스 영역
    "frame_rate": 30,              # 프레임 레이트
    "half_precision": True,        # 반정밀도 사용
    "device": "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "cpu"
}

# ReID 설정
REID_CONFIG = {
    "model_path": "models/weights/yolo11m.pt",
    "device": "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "cpu",
    "batch_size": 32,
    "num_features": 2048,
    "max_features_per_id": 10,
    "similarity_threshold": 0.5,         # 일반 유사도 임계값 (더 관대한 매칭용)
    "high_similarity_threshold": 0.7,    # 높은 유사도 임계값 (정확한 매칭용)
    "min_matching_features": 2,          # 최소 매칭되어야 할 feature 개수
    "enable_cross_camera_matching": True,  # 크로스 카메라 매칭 활성화
    "cross_camera_threshold": 0.6,       # 크로스 카메라 매칭 임계값
    "enable_pre_registration": True,     # 사전 등록 매칭 활성화
    "pre_registration_threshold": 0.65,  # 사전 등록 매칭 임계값
    "enable_matching_cache": True,       # 매칭 캐시 사용 여부
    "cache_stats_interval": 10,          # 캐시 통계 출력 간격 (프레임 단위)
}

# 사전 등록 매칭 설정
PRE_REGISTRATION_CONFIG = {
    "similarity_threshold": 0.65,     # 사전 등록 매칭용 유사도 임계값 (높은 정확도 필요)
    "min_matching_features": 2,      # 최소 매칭되어야 할 feature 개수 (1~5 권장)
    "max_features_per_id": 10,       # Global ID당 최대 feature 개수 (고정값)
    "enable_matching_cache": True,   # 매칭 캐시 사용 여부
    "cache_stats_interval": 10,      # 캐시 통계 출력 간격 (프레임 단위)
}

# 바운딩 박스 안정화 기반 매칭 설정 (새로 추가)
BOUNDING_BOX_MATCHING_CONFIG = {
    "min_frames_for_stability": 5,      # 안정화 판단을 위한 최소 프레임 수
    "stability_threshold": 0.05,         # 크기 변화 허용 임계값 (10%)
    "min_bbox_area": 20000,              # 최소 바운딩 박스 크기 (픽셀)
    "max_tracking_frames": 10,          # 추적할 최대 프레임 수
    "enable_bbox_tracking": True,       # 바운딩 박스 추적 활성화
    "debug_bbox_tracking": True,        # 바운딩 박스 추적 디버깅 출력
}

# 향상된 유사도 측정 설정 (새로 추가)
SIMILARITY_CONFIG = {
    "method": "ensemble",               # 유사도 측정 방법: "cosine", "euclidean", "ensemble"
    "threshold": 0.6,                   # 유사도 임계값 (0.0~1.0, 높을수록 더 엄격)
    "ensemble_weights": [0.95, 0.5],     # 앙상블 가중치 [코사인, 유클리드]
    "enable_detailed_logging": True,    # 상세 로깅 활성화
    "fallback_threshold": 0.4,          # 폴백 임계값 (매칭 실패 시 더 낮은 임계값으로 재시도)
}

# 호모그래피 매트릭스 (카메라별)
HOMOGRAPHY_MATRICES = {
    0: [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ],
    1: [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]
}

# 백엔드 설정
BACKEND_CONFIG = {
    "url": "http://localhost:3000",
    "timeout": 5,
    "retry_count": 3
}

# Redis 설정
REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "password": None,
    "decode_responses": True
}

# 로깅 설정
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "tracking_system.log"
}

# 성능 설정
PERFORMANCE_CONFIG = {
    "max_fps": 30,
    "frame_skip": 1,
    "enable_gpu": True,
    "enable_half_precision": True,
    "batch_processing": True
}
