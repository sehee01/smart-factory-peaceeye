# Smart Factory PeaceEye App

스마트 팩토리 환경에서 객체 탐지, 추적, Re-Identification을 통합한 모니터링 시스템

## 프로젝트 구조

```
app/
├── detector/                      ← ByteTrack 탐지 책임
│   ├── detector_manager.py        # ByteTrackDetectorManager
│   └── bytetrack_processor.py     # YOLO + BYTETracker 실행
│
├── reid/                          ← ReID 책임
│   ├── reid_manager.py            # GlobalReIDManager
│   ├── redis_handler.py           # FeatureStoreRedisHandler
│   ├── similarity.py              # FeatureSimilarityCalculator
│   ├── pre_registration.py        # 사전 등록 기능
│   └── models/                    # ReID 모델
│       └── weights/               # ReID 모델 가중치
│
├── models/                        ← 모델 및 매핑 관련
│   ├── main.py                    # 모델 초기화
│   ├── requirements.txt           # 모델 의존성
│   ├── weights/                   # YOLO 모델 가중치
│   ├── yolo/                      # YOLO 관련 모듈
│   ├── result/                    # 결과 저장
│   └── mapping/                   ← 좌표 변환
│       ├── homography_calibration.py  # 호모그래피 보정
│       ├── point_transformer.py       # 좌표 변환
│       └── 픽셀추출_실행파일.py       # 픽셀 추출 도구
│
├── config/
│   └── settings.py                # Thresholds, Redis conf 등 설정 관리
│
├── pre_img/                       ← 사전 등록 이미지
│   ├── 10/                        # Global ID 10번 이미지들
│   ├── 11/                        # Global ID 11번 이미지들
│   └── 12/                        # Global ID 12번 이미지들
│
├── test_video/                    ← 테스트 비디오 파일들
│   ├── final01.mp4, final02.mp4   # 최종 테스트 비디오
│   ├── PPE01.mp4, PPE02.mp4       # PPE 테스트
│   ├── KSEB01.mp4 ~ KSEB03.mp4    # KSEB 테스트
│   ├── globaltest01.mp4 ~ globaltest04.mp4  # 글로벌 테스트
│   └── 0_te2.mp4, 0_te3.mp4       # 기타 테스트
│
├── main.py                        # AppOrchestrator (멀티스레딩 지원)
├── debug_main.py                  # 디버그용 메인 (단일 스레드)
├── pre_registration_test.py       # 사전 등록 테스트
├── image_processor.py             # 이미지 처리 및 feature 추출
├── requirements.txt               # 프로젝트 의존성
├── docker-compose.yml             # Redis 컨테이너 설정
└── README.md                      # 이 파일
```

## 주요 컴포넌트

### 🔍 **Detector (객체 탐지)**
- **`detector_manager.py`**: ByteTrack 탐지 관리자
- **`bytetrack_processor.py`**: YOLO + ByteTracker 실행

### 🆔 **ReID (재식별)**
- **`reid_manager.py`**: 글로벌 ReID 관리
- **`redis_handler.py`**: Redis 데이터 저장/조회
- **`similarity.py`**: 특징 유사도 계산
- **`pre_registration.py`**: 사전 등록 기능

### 🗺️ **Mapping (좌표 변환)**
- **`homography_calibration.py`**: 호모그래피 보정
- **`point_transformer.py`**: 이미지 좌표 → 실제 좌표 변환
- **`픽셀추출_실행파일.py`**: 픽셀 추출 도구

### 🖼️ **Image Processing**
- **`image_processor.py`**: 이미지 처리 및 feature 추출

### ⚙️ **Configuration**
- **`settings.py`**: 임계값, Redis 설정, GUI 설정 등

## 실행 방법

### 1. Redis 서버 시작
```bash
# Docker 사용
docker-compose up -d

# Podman 사용
podman compose up -d
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 애플리케이션 실행

#### 멀티스레딩 (권장)
```bash
python3 main.py --multi_thread
```

#### 디버그 모드 (단일 스레드)
```bash
python3 debug_main.py
```

#### 사전 등록 테스트
```bash
python3 pre_registration_test.py
```

## Redis 관리

### Redis 서버 중지
```bash
# Docker 사용
docker-compose down

# Podman 사용
podman compose down
```

### Redis 상태 확인
```bash
# Docker 사용
docker-compose ps

# Podman 사용
podman compose ps
```

## 기능

- **멀티 카메라 지원**: 여러 비디오 동시 처리
- **글로벌 ReID**: Redis 기반 객체 재식별
- **사전 등록**: 특정 객체 미리 등록 가능
- **좌표 변환**: 이미지 좌표 → 실제 공간 좌표
- **실시간 추적**: ByteTrack 기반 객체 추적
- **Redis 캐싱**: 특징 벡터 및 메타데이터 저장

## 테스트 데이터

- **`test_video/`**: 다양한 테스트 비디오 파일
- **`pre_img/`**: 사전 등록용 이미지 (Global ID별 폴더)
- **`models/weights/`**: 훈련된 YOLO 모델
- **`reid/models/weights/`**: ReID 모델 가중치