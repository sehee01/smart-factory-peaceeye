프로젝트 구조 리팩터링

app/
├── detector/                      ← ByteTrack 탐지 책임
│   ├── detector_manager.py        # ByteTrackDetectorManager
│   └── bytetrack_processor.py     # YOLO + BYTETracker 실행
│
├── reid/                          ← ReID 책임
│   ├── reid_manager.py            # GlobalReIDManager
│   ├── redis_handler.py           # FeatureStoreRedisHandler
│   └── similarity.py              # FeatureSimilarityCalculator
│
├── main.py                        # AppOrchestrator (stream 처리 루프)
│
├── config/
│   └── settings.py                # Thresholds, Redis conf 등 설정 관리
│
├── tests/                         # 유닛 테스트
└── README.md


## Redis 설정

### Redis 서버 시작
```bash
# Docker 사용
docker-compose up -d

# Podman 사용
podman compose up -d
```

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

### 시작
```bash
python3 main.py --multi_thread
```