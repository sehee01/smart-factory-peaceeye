# 📊 PerformanceLogger 간단 사용법

## 🎯 **개요**

`PerformanceLogger`는 객체 인식 프로젝트의 성능을 측정하고 CSV 파일로 저장하는 간소화된 클래스입니다.

## 🚀 **기본 사용법**

### **1. 초기화**
```python
from result.performance_logger import PerformanceLogger

# CSV 파일로 성능 데이터 저장
logger = PerformanceLogger("result")
```

### **2. 성능 측정**
```python
# 프레임별 타이밍 시작
logger.start_frame_timing(frame_id, camera_id)

# 탐지 타이밍
logger.start_detection_timing()
# ... 탐지 작업 ...
logger.end_detection_timing()

# 트래킹 타이밍
logger.start_tracking_timing()
# ... 트래킹 작업 ...
logger.end_tracking_timing()

# 사전 등록 매칭 타이밍 (선택적)
logger.start_pre_match_timing()
# ... 사전 등록 매칭 작업 ...
logger.end_pre_match_timing()

# ReID 타이밍 (선택적)
logger.start_same_camera_reid_timing()
# ... 같은 카메라 ReID 작업 ...
logger.end_same_camera_reid_timing()

logger.start_cross_camera_reid_timing()
# ... 다른 카메라 ReID 작업 ...
logger.end_cross_camera_reid_timing()

# 객체 수 설정
logger.set_object_count(len(detected_objects))

# 성능 데이터 저장
logger.log_frame_performance()
```

### **3. 결과 확인**
```python
# 성능 요약 출력
logger.print_summary()
```

## 📁 **출력 파일**

- `performance_log_{timestamp}.csv` - 프레임별 상세 데이터

## 🔧 **CSV 구조**

### **헤더**
```
frame_id, object_count, detection_time_ms, tracking_time_ms, pre_match_time_ms, same_camera_time_ms, cross_camera_time_ms, total_time_ms
```

### **데이터 예시**
```
1, 5, 25.3, 12.1, 0.0, 3.2, 0.0, 40.6
2, 3, 18.7, 8.9, 0.0, 2.1, 0.0, 29.7
3, 4, 22.1, 10.5, 5.2, 0.0, 0.0, 37.8
```

## 💡 **사용 예시**

### **완전한 예시**
```python
from result.performance_logger import PerformanceLogger
import time

# 로거 초기화
logger = PerformanceLogger("result")

# 프레임 처리
for frame_id in range(1, 101):
    # 타이밍 시작
    logger.start_frame_timing(frame_id, 0)
    
    # 탐지
    logger.start_detection_timing()
    time.sleep(0.02)  # 탐지 작업 시뮬레이션
    logger.end_detection_timing()
    
    # 트래킹
    logger.start_tracking_timing()
    time.sleep(0.01)  # 트래킹 작업 시뮬레이션
    logger.end_tracking_timing()
    
    # 사전 등록 매칭 (선택적)
    if frame_id % 10 == 0:  # 10프레임마다 실행
        logger.start_pre_match_timing()
        time.sleep(0.005)  # 사전 등록 매칭 작업 시뮬레이션
        logger.end_pre_match_timing()
    
    # ReID (선택적)
    if frame_id % 5 == 0:  # 5프레임마다 실행
        logger.start_same_camera_reid_timing()
        time.sleep(0.003)  # ReID 작업 시뮬레이션
        logger.end_same_camera_reid_timing()
    
    # 객체 수 설정
    logger.set_object_count(5)
    
    # 성능 데이터 저장
    logger.log_frame_performance()

# 결과 확인
logger.print_summary()
```

## 📊 **콘솔 출력 예시**

```
📊 성능 요약 (총 100 프레임)
============================================================
총 객체 수: 500
평균 탐지 시간: 22.0ms
평균 트래킹 시간: 10.5ms
평균 총 시간: 35.2ms

📊 ReID 통계 (0값 제외)
----------------------------------------
pre_match      : 15 프레임, 객체  75개, 평균   5.2ms
same_camera    : 80 프레임, 객체 400개, 평균   3.1ms
cross_camera   :  5 프레임, 객체  25개, 평균   8.7ms

📁 CSV 파일 저장 완료: result/performance_log_20250113_144612.csv
```

## ⚠️ **주의사항**

1. **CSV 파일**: 모든 데이터는 CSV 파일에 실시간으로 저장됩니다
2. **0값 처리**: ReID 통계에서는 0값을 제외하고 평균 계산합니다
3. **파일 위치**: `{output_dir}/` 폴더에 CSV 파일이 생성됩니다

## 🎉 **장점**

- **간단함**: 복잡한 설정 없이 바로 사용 가능
- **빠름**: CSV 파일에 즉시 저장되어 빠른 처리
- **호환성**: 모든 프로그램에서 CSV 파일 열기 가능
- **가독성**: 코드가 간결하고 이해하기 쉬움

## 🔍 **데이터 구조**

### **컬럼 설명**
- `frame_id`: 프레임 번호
- `object_count`: 탐지된 객체 수
- `detection_time_ms`: 탐지에 소요된 시간 (밀리초)
- `tracking_time_ms`: 트래킹에 소요된 시간 (밀리초)
- `pre_match_time_ms`: 사전 등록 매칭에 소요된 시간 (밀리초)
- `same_camera_time_ms`: 같은 카메라 ReID에 소요된 시간 (밀리초)
- `cross_camera_time_ms`: 다른 카메라 ReID에 소요된 시간 (밀리초)
- `total_time_ms`: 전체 처리 시간 (밀리초)

**결론**: 이제 **CSV 파일로만 결과를 저장하는 간단하고 효율적인** PerformanceLogger를 사용할 수 있습니다! 🚀
