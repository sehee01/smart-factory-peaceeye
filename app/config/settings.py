# config/settings.py

# YOLO 모델 경로 (test_ultralytics_tracking.py와 동일하게 설정)
YOLO_MODEL_PATH = "models/weights/yolo11m.pt"  # test_ultralytics_tracking.py와 동일한 모델
PRE_REGISTER_MODEL_PATH = "models/weights/yolo11m.pt"
PPE_MODEL_PATH = "models/weights/best_yolo11n.pt"
# BYTETracker 설정 (더 엄격한 탐지 임계값 적용)
TRACKER_CONFIG = {
     "track_thresh": 0.6,                    # 탐지 신뢰도 임계값 (0.5 → 0.6으로 더 엄격하게)
     "match_thresh": 0.9,                    # Ultralytics 기본값: 0.9 (IOU 매칭 임계값)
    "track_buffer": 60,                     # Ultralytics 기본값: 30 (트랙 유지 프레임 수)
     "mot20": False,                         # Ultralytics 기본값: False (MOT20 데이터셋 사용 여부)
    "frame_rate": 30,                       # Ultralytics 기본값: 30 (프레임 레이트)
    "target_width": 640,                    # 프레임 리사이즈 목표 너비
     "aspect_ratio_thresh": 1.6,            # Ultralytics 기본값: 1.6 (종횡비 임계값)
     "min_box_area": 100,                   # Ultralytics 기본값: 100 (최소 박스 면적)
}

# Redis 설정 (글로벌 연결 설정)
REDIS_CONFIG = {
    "host": "localhost",              # Redis 서버 호스트
    "port": 6379                     # Redis 서버 포트
}

# ReID 설정 (원본의 복잡한 설정 반영)
REID_CONFIG = {
    # 기본 유사도 임계값 (같은/다른 카메라 매칭용)
    "threshold": 0.9,                    # 기본 유사도 임계값 (0.7~0.9 권장)
    "ttl": 300,                          # 기본 TTL (초 단위)
    "frame_rate": 15,                    # 프레임 레이트 (FPS)
    "feature_ttl": 3000,                 # Feature TTL (프레임 단위, 100초)
    "similarity_threshold": 0.5,         # 일반 유사도 임계값 (더 관대한 매칭용)
    
    # 사전 등록 매칭 설정
    "pre_registration": {
        "similarity_threshold": 0.65,     # 사전 등록 매칭용 유사도 임계값 (높은 정확도 필요)
        "min_matching_features": 2,      # 최소 매칭되어야 할 feature 개수 (1~5 권장)
        "max_features_per_id": 10,       # Global ID당 최대 feature 개수 (고정값)
        "enable_matching_cache": True,   # 매칭 캐시 사용 여부
        "cache_stats_interval": 10,      # 캐시 통계 출력 간격 (프레임 단위)
    },
    
    # 같은 카메라 내 매칭 설정
    "same_camera": {
        "location_threshold": 0.05,      # 위치 기반 필터링 임계값 (0.05~0.1 권장)
        "max_distance": 200,             # 바운딩 박스 중심점 간 최대 거리 (픽셀)
        "dynamic_threshold_factor": 0.7, # 위치 기반 동적 임계값 조정 계수 (기본 임계값에 곱해짐)
                                        # 계산식: threshold * (1.0 - location_score * 0.7)
                                        # 예: 위치가 가까우면 (location_score=0.8) → 0.8 * (1.0 - 0.8 * 0.7) = 0.352
        "min_threshold_factor": 0.2,     # 최소 임계값 보장 계수 (기본 임계값에 곱해짐)
                                        # 계산식: max(dynamic_threshold, threshold * 0.2)
                                        # 예: 0.8 * 0.2 = 0.16 (최소 보장 임계값)
        "weight_start": 0.5,             # 가중 평균 계산 시작 가중치 (최근 feature에 더 높은 가중치)
        "weight_end": 1.0,               # 가중 평균 계산 끝 가중치 (최근 feature에 더 높은 가중치)
    },
    
    # 다른 카메라간 매칭 설정
    "cross_camera": {
        "threshold_cross": 0.8,     # 다른 카메라 매칭 임계값 
                                        # 계산식: threshold * 1.0 = 0.8
                                        # 더 엄격하게 하려면: 1.2 → 0.8 * 1.2 = 0.96
                                        # 더 관대하게 하려면: 0.8 → 0.8 * 0.8 = 0.64
        "weight_start": 0.5,             # 가중 평균 계산 시작 가중치 (최근 feature에 더 높은 가중치)
        "weight_end": 1.0,               # 가중 평균 계산 끝 가중치 (최근 feature에 더 높은 가중치)
    },
    
    # ReID 데이터별 TTL 설정 (글로벌 Redis 연결은 REDIS_CONFIG 사용)
    "redis": {
        "feature_ttl": 300,              # Feature 저장 TTL (초) - 기본 feature 데이터
        "track_ttl": 600,                # Track 정보 TTL (초) - track 메타데이터
        "pre_registration_ttl": 0,      # 사전 등록 데이터 TTL (초, 0=무제한)
        "max_features_per_track": 10,    # Track당 최대 feature 개수
        "cleanup_buffer": 2,             # 만료된 트랙 정리시 TTL 배수 (TTL * 2)
    },
    
    # 성능 및 정확도 설정
    "performance": {
        "use_weighted_average": True,    # 가중 평균 feature 사용 여부
        "enable_location_filtering": True, # 위치 기반 필터링 사용 여부
        "enable_dynamic_threshold": True, # 동적 임계값 사용 여부
        "cache_similarity_results": False, # 유사도 계산 결과 캐싱 여부
    }
}

# 입력 비디오 경로 (원본과 동일)
VIDEO_INPUT_PATHS = [
    "test_video/TEST100.mp4",
     "test_video/TEST1.mp4"
]

# 카메라별 호모그래피 매트릭스 (Ground Truth 캘리브레이션)
HOMOGRAPHY_MATRICES = {
    0: [  # 카메라 0 (TEST100)
        [
            -0.7054058165407032,
            -9.175014931008594,
            360.7030953630252
        ],
        [
            3.639540004543024,
            6.908272367795101,
            -3448.843643339362
        ],
        [
            -8.88526112076813e-05,
            -0.0031670698606516006,
            1.0
        ]
    ],
    1: [  # 카메라 1 (TEST1)
        
       [
            4.1963912058589194,
            -42.47203313918037,
            761.2894933693545
        ],
        [
            -0.911343564699018,
            -30.168272520102708,
            14578.643632850666
        ],
        [
            -7.233425744493245e-05,
            -0.007065025908464806,
            1.0
        ]
  
    ]
}

# Ground Truth 캘리브레이션 파일 경로
CALIBRATION_FILES = {
    0: "TEST100_ground_truth_calibration.json",
    1: "TEST1_ground_truth_calibration.json"
}

# 기존 호모그래피 매트릭스 (하위 호환성을 위해 유지)
HOMOGRAPHY_MATRIX = HOMOGRAPHY_MATRICES[0]  # 카메라 0의 매트릭스를 기본값으로 사용

# 탐지 설정
DETECTION_CONFIG = {
    "person_classes": ["person", "saram"],  # 탐지할 클래스들
    "confidence_threshold": 0.5,
    "nms_threshold": 0.4,
}

# 멀티스레딩 설정 (원본과 동일)
MULTITHREADING_CONFIG = {
    "enabled": True,
    "max_frames": 3000,  # 최대 처리 프레임 수 (약 10초)
    "frame_timeout": 0.1,  # 프레임 큐 타임아웃
    "thread_timeout": 1.0,  # 스레드 종료 타임아웃
}

# GUI 설정 (원본과 동일)
GUI_CONFIG = {
    "window_width": 1920,
    "window_height": 1080,
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
    "level": "DEBUG",  # INFO, DEBUG, WARNING, ERROR 중 선택
    "show_debug": True,
    "show_coordinates": True,
    "show_reid_matches": True,
    "show_disappearing_objects": True,
    "reid_detailed_logging": True,  # ReID 매칭 과정 상세 로깅
    "similarity_detailed_logging": True,  # 유사도 계산 상세 로깅
}

# 성능 설정 (원본과 동일)
PERFORMANCE_CONFIG = {
    "use_gpu": True,
    "half_precision": True,  # GPU 사용시 half precision
    "batch_size": 1,
    "num_workers": 0,  # 멀티스레딩에서는 0으로 설정
}
