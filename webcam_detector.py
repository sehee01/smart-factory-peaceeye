
import cv2
import json
import torch
import sys
import argparse
import numpy as np
from ultralytics import YOLO

# ByteTrack 경로 추가
sys.path.append('ByteTrack')
from yolox.tracker.byte_tracker import BYTETracker

# np.float 에러 방지 (호환성 유지)
if not hasattr(np, 'float'):
    np.float = float

# 모델 로드
try:
    model = YOLO('models/weights/best.pt')
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the model weight file exists at 'models/weight/best.pt'")
    sys.exit(1)

classNames = model.names

# 웹캠 열기
# capture = cv2.VideoCapture(0)
video_path = "test_vedio/test01.mp4"
capture = cv2.VideoCapture(video_path)
if not capture.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)

frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Webcam resolution: {frame_width}x{frame_height}")

# ByteTrack 설정
tracker_args = argparse.Namespace(
    track_thresh=0.5,  #감지 신뢰도 - 이상만 출력
    match_thresh=0.8,   #IOU - 이상만 같은 객체로 매칭
    track_buffer=60,    # -프레임까지 끊겨도 유지
    mot20=False
)
tracker = BYTETracker(tracker_args, frame_rate=30)

tracked_ids = set()
person_count = 0

while True:
    ret, frame = capture.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # 객체 탐지
    detection = model(frame)[0]
    dets = []  # 사람일 경우 추가되는 list

    for box in detection.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        # 'person' 또는 'persona' 클래스 ID를 확인하고 신뢰도가 0.5 이상인 경우에만 처리
        if classNames[cls_id].lower() in ['person', 'persona'] and conf > 0.5:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dets.append([x1, y1, x2, y2, conf])

    # 트래커 업데이트 및 JSON 출력
    frame_detections_json = []
    if len(dets) > 0:
        online_targets = tracker.update(torch.tensor(dets), (frame_height, frame_width), (frame_height, frame_width))
        for t in online_targets:
            tid = t.track_id
            if tid not in tracked_ids:
                tracked_ids.add(tid)
                person_count += 1
            
            xmin, ymin, xmax, ymax = map(int, t.tlbr)

            # JSON 데이터 생성
            detection_data = {
                "track_id": int(tid),
                "class": "person",
                "confidence": float(t.score),
                "bbox_xyxy": [xmin, ymin, xmax, ymax]
            }
            frame_detections_json.append(detection_data)

            # 바운딩 박스 그리기
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # ID 및 클래스 이름 표시
            label = f'Person_ID:{int(tid)}'
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 현재 프레임의 탐지 결과 JSON으로 출력
    if frame_detections_json:
        print(json.dumps(frame_detections_json, indent=4))

    # 사람 수 표시
    cv2.putText(frame, f"Total Persons: {person_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # 화면에 프레임 표시
    cv2.imshow("Webcam Object Tracking", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# 자원 해제
capture.release()
cv2.destroyAllWindows()
print("Webcam feed stopped.")
