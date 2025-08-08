# 제한구역 좌표 설정 도구 사용법

## 개요
`zone_coordinate_picker.py`는 비디오 파일을 띄워서 마우스로 직접 제한구역 좌표를 찍어서 JSON 파일로 저장하는 도구입니다.

## 설치 요구사항
```bash
pip install opencv-python numpy
```

## 사용 방법

### 1. 제한구역 좌표 설정하기

```bash
# 기본 사용법 (main2.py와 동일한 비디오 파일 사용)
python zone_coordinate_picker.py

# 특정 비디오 파일 지정
python zone_coordinate_picker.py ../test_video/KSEB03.mp4

# 사용 가능한 비디오 파일 목록 확인
python zone_coordinate_picker.py --list-videos

# 기존 JSON 파일에서 좌표 로드
python zone_coordinate_picker.py --load zone_coordinates_20241201_143022.json
```

### 2. 실행 후 조작법

1. **비디오 창이 열리면 마우스로 드래그하여 제한구역을 그립니다**
   - 마우스 왼쪽 버튼을 누른 상태로 드래그
   - 사각형 모양으로 제한구역 설정

2. **키보드 단축키**
   - `R`: 모든 좌표 재설정
   - `S`: 현재 좌표를 JSON 파일로 저장
   - `L`: 기존 JSON 파일에서 좌표 로드
   - `Q` 또는 `ESC`: 종료

3. **화면 표시**
   - 빨간색 사각형: 설정된 제한구역
   - 파란색 원: 감지 반경 (기본 100픽셀)
   - 초록색 사각형: 현재 그리는 중인 구역

### 3. JSON 파일에서 설정 파일 생성

```bash
# JSON 파일에서 Python 설정 파일 생성
python zone_coordinate_picker.py --create-config zone_coordinates_20241201_143022.json
```

## 파일 구조

### 생성되는 JSON 파일 예시
```json
{
  "video_path": "../test_video/KSEB03.mp4",
  "created_at": "2024-12-01T14:30:22.123456",
  "zones": [
    {
      "x1": 800,
      "y1": 400,
      "x2": 1120,
      "y2": 680,
      "threshold": 100,
      "zone_name": "restricted_zone",
      "created_at": "2024-12-01T14:30:22.123456"
    }
  ],
  "total_zones": 1
}
```

### 생성되는 설정 파일 예시
```python
# 제한구역 설정 파일 (자동 생성)
RESTRICTED_ZONE_EXAMPLES = {
    'zone_1': {
        'x1': 800, 'y1': 400,
        'x2': 1120, 'y2': 680,
        'threshold': 100
    },
}
```

## main2.py와의 연동

1. **JSON 파일 자동 로드**: `main2.py`는 실행 시 자동으로 가장 최근의 `zone_coordinates_*.json` 파일을 찾아서 로드합니다.

2. **우선순위**:
   - 1순위: JSON 파일 (`zone_coordinates_*.json`)
   - 2순위: 설정 파일 (`restricted_zone_config.py`)
   - 3순위: 기본 설정

3. **실행 순서**:
    ```bash
    # 1. 사용 가능한 비디오 파일 확인 (선택사항)
    python zone_coordinate_picker.py --list-videos
    
    # 2. 제한구역 좌표 설정 (main2.py와 동일한 비디오 사용)
    python zone_coordinate_picker.py
    
    # 3. main2.py 실행 (자동으로 JSON 파일 로드)
    python main2.py
    ```

## 설정 옵션

### 감지 반경 조정
- 기본값: 100픽셀
- `zone_coordinate_picker.py`의 `self.threshold` 값을 수정하여 변경 가능

### 알람 임계값 조정
- `main2.py`의 `RESTRICTED_ZONE_CONFIG`에서 조정:
  - `alarm_threshold`: 0 (제한구역 내부 진입 시 치명적 알람)
  - `warning_threshold`: 100 (1미터 이내 접근 시 경고 알람)

## 문제 해결

### 1. 비디오 파일을 찾을 수 없는 경우
```bash
# 절대 경로 사용
python zone_coordinate_picker.py "C:/path/to/your/video.mp4"
```

### 2. OpenCV 창이 열리지 않는 경우
- OpenCV 설치 확인: `pip install opencv-python`
- 디스플레이 드라이버 확인

### 3. JSON 파일이 로드되지 않는 경우
- 파일명이 `zone_coordinates_`로 시작하는지 확인
- 파일이 `main2.py`와 같은 디렉토리에 있는지 확인

## 팁

1. **정확한 좌표 설정**: 비디오의 첫 번째 프레임에서 설정하므로, 원하는 장면이 나올 때까지 비디오를 재생한 후 일시정지하고 좌표를 설정하세요.

2. **여러 제한구역**: 여러 번 드래그하여 여러 개의 제한구역을 설정할 수 있습니다.

3. **좌표 확인**: 설정 후 콘솔에 출력되는 좌표 정보를 확인하세요.

4. **백업**: 중요한 설정은 JSON 파일을 백업해두세요.
