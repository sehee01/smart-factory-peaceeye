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
├── app/
│   └── main.py                    # AppOrchestrator (stream 처리 루프)
│
├── config/
│   └── settings.py                # Thresholds, Redis conf 등 설정 관리
│
├── tests/                         # 유닛 테스트
└── README.md


python -m main
