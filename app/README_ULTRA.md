# Smart Factory PeaceEye - Ultralytics Tracking Version

ByteTrack 대신 **Ultralytics의 내장 tracking 기능**을 사용하는 버전입니다.

## 🆕 새로운 파일들

### Core Files
- **`detector/ultralytics_tracker.py`**: Ultralytics 내장 tracking을 사용하는 detector manager
- **`integrated_tracking_system_ultra.py`**: Ultralytics tracking을 사용하는 통합 시스템
- **`new_main_ultra.py`**: Ultralytics 버전 메인 실행 파일
- **`test_ultralytics_tracking.py`**: Ultralytics tracking 기능 테스트 스크립트

## 🚀 실행 방법

### 1. 기본 실행
```bash
# Ultralytics tracking 버전 실행
python new_main_ultra.py

# 특정 비디오 파일 지정
python new_main_ultra.py --videos test_video/final01.mp4 test_video/final02.mp4

# 다른 YOLO 모델 사용
python new_main_ultra.py --yolo_model models/weights/yolov8n.pt
```

### 2. 테스트 실행
```bash
# Ultralytics tracking 기능 테스트
python test_ultralytics_tracking.py
```

### 3. 원본 ByteTrack 버전과 비교
```bash
# 원본 ByteTrack 버전
python new_main.py

# 새로운 Ultralytics 버전
python new_main_ultra.py
```

## 🔧 주요 차이점

### ByteTrack vs Ultralytics Tracking

| 기능 | ByteTrack | Ultralytics Tracking |
|------|-----------|---------------------|
| **의존성** | 외부 ByteTrack 폴더 필요 | Ultralytics 내장 |
| **설치** | 복잡한 설치 과정 | `pip install ultralytics` |
| **성능** | 최적화된 성능 | 내장 최적화 |
| **유지보수** | 별도 관리 필요 | Ultralytics와 함께 업데이트 |
| **메모리** | 추가 메모리 사용 | 통합 메모리 관리 |

### 장점
- ✅ **간단한 설치**: ByteTrack 폴더 불필요
- ✅ **통합 관리**: Ultralytics와 함께 업데이트
- ✅ **메모리 효율**: 추가 의존성 없음
- ✅ **호환성**: 최신 Ultralytics 버전과 호환

### 단점
- ⚠️ **제한된 커스터마이징**: ByteTrack만큼 세밀한 설정 불가
- ⚠️ **성능 차이**: 특정 상황에서 ByteTrack보다 성능이 낮을 수 있음

## 📊 성능 비교

### 테스트 환경
- **모델**: `models/weights/bestcctv.pt`
- **비디오**: `test_video/final01.mp4`
- **프레임 수**: 1000 프레임

### 결과 예시
```
ByteTrack Version:
- FPS: ~25-30
- Memory: ~2.5GB
- Track ID Consistency: High

Ultralytics Version:
- FPS: ~20-25
- Memory: ~2.0GB
- Track ID Consistency: Good
```

## 🛠️ 설정

### Tracker 설정 (`config/settings.py`)
```python
TRACKER_CONFIG = {
    "target_width": 640,        # 입력 프레임 너비
    "track_buffer": 30,         # 추적 버퍼 크기
    "frame_rate": 30,           # 프레임 레이트
}
```

### Ultralytics 내장 설정
```python
# ultralytics_tracker.py에서 사용
tracker="bytetrack.yaml"  # Ultralytics의 ByteTrack 구현
persist=True              # 프레임 간 추적 상태 유지
```

## 🔍 디버깅

### 로그 확인
```bash
# 상세 로그 출력
python new_main_ultra.py --videos test_video/final01.mp4 2>&1 | tee ultra_log.txt
```

### 성능 모니터링
```bash
# 성능 요약 확인
python new_main_ultra.py
# 결과 파일: tracking_results_ultra.json
```

## 📁 파일 구조

```
app/
├── detector/
│   ├── ultralytics_tracker.py          # 🆕 Ultralytics Tracker
│   ├── detector_manager.py             # 원본 ByteTrack Manager
│   └── bytetrack_processor.py          # 원본 ByteTrack Processor
├── integrated_tracking_system_ultra.py # 🆕 Ultralytics 통합 시스템
├── new_main_ultra.py                   # 🆕 Ultralytics 메인
├── test_ultralytics_tracking.py        # 🆕 테스트 스크립트
├── new_main.py                         # 원본 ByteTrack 메인
└── integrated_tracking_system.py       # 원본 ByteTrack 통합 시스템
```

## 🎯 사용 권장사항

### Ultralytics Tracking 사용 시기
- ✅ **빠른 프로토타이핑**이 필요할 때
- ✅ **ByteTrack 설치 문제**가 있을 때
- ✅ **메모리 제약**이 있는 환경
- ✅ **간단한 추적**만 필요한 경우

### ByteTrack 사용 시기
- ✅ **최고 성능**이 필요할 때
- ✅ **복잡한 추적 시나리오**가 있을 때
- ✅ **세밀한 설정**이 필요할 때
- ✅ **연구/개발** 목적

## 🚨 주의사항

1. **모델 호환성**: YOLOv8 모델만 지원 (YOLOv5는 제한적)
2. **메모리 사용**: GPU 메모리 사용량이 다를 수 있음
3. **성능 차이**: 환경에 따라 성능 차이가 있을 수 있음
4. **설정 제한**: ByteTrack만큼 세밀한 설정이 불가능

## 📞 지원

문제가 발생하면 다음을 확인하세요:

1. **Ultralytics 버전**: `pip show ultralytics`
2. **CUDA 지원**: `python -c "import torch; print(torch.cuda.is_available())"`
3. **모델 파일**: `models/weights/bestcctv.pt` 존재 확인
4. **비디오 파일**: `test_video/final01.mp4` 존재 확인

## 🔄 마이그레이션

### ByteTrack에서 Ultralytics로
```bash
# 1. 기존 코드 백업
cp new_main.py new_main_bytetrack_backup.py

# 2. 새로운 Ultralytics 버전 사용
python new_main_ultra.py

# 3. 성능 비교
python new_main.py      # ByteTrack
python new_main_ultra.py # Ultralytics
```

### Ultralytics에서 ByteTrack로
```bash
# 원본 ByteTrack 버전 사용
python new_main.py
```
