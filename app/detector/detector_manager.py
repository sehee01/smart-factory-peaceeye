import cv2
from ultralytics import YOLO
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker


class ByteTrackDetectorManager:
    """
    ByteTrack 기반 객체 탐지 및 추적을 통합 관리하는 매니저 클래스
    - YOLO를 통한 객체 탐지
    - BYTETracker를 통한 추적 유지 및 ID 할당
    """

    def __init__(self, model_path: str, tracker_config: dict):
        """
        :param model_path: YOLO 모델 경로 (e.g., yolov8n.pt)
        :param tracker_config: BYTETracker 설정 (dict 형태)
        """
        self.model = YOLO(model_path)
        self.tracker = BYTETracker(tracker_config)

    def detect_and_track(self, frame, frame_id: int, fps: float = 30.0):
        """
        주어진 프레임에서 객체 탐지 및 추적 실행

        :param frame: OpenCV 이미지 프레임
        :param frame_id: 현재 프레임 번호 (int)
        :param fps: 현재 FPS (기본 30.0)
        :return: track_list - 각 트랙에 대한 dict (id, bbox, score 등)
        """
        # 1. YOLO 객체 탐지
        results = self.model(frame)[0]  # first batch element
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = box.conf[0].item()
            cls = int(box.cls[0].item())
            detections.append([x1, y1, x2 - x1, y2 - y1, score, cls])

        # 2. BYTETracker 입력 준비 (nparray: [x, y, w, h, score])
        import numpy as np
        dets = np.array(detections, dtype=np.float32)
        tracked_objs = self.tracker.update(dets, frame, frame_id, fps)

        # 3. 추적 결과 정제
        track_list = []
        for track in tracked_objs:
            if not track.is_activated:
                continue
            x1, y1, x2, y2 = track.tlbr  # top-left bottom-right
            track_list.append({
                "track_id": track.track_id,
                "bbox": [x1, y1, x2, y2],
                "score": track.score,
                "cls": track.cls,
            })

        return track_list
