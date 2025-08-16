# bytetrack_processor.py
import numpy as np
from types import SimpleNamespace
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

class ByteTrackProcessor:
    def __init__(self, tracker_config: dict):
        # Ultralytics ByteTrack 기본값과 동일하게 설정
        args = SimpleNamespace(
            track_thresh=tracker_config.get("track_thresh", 0.6),      # Ultralytics 기본값: 0.6
            match_thresh=tracker_config.get("match_thresh", 0.9),      # Ultralytics 기본값: 0.9
            track_buffer=tracker_config.get("track_buffer", 30),       # Ultralytics 기본값: 30
            aspect_ratio_thresh=tracker_config.get("aspect_ratio_thresh", 1.6),  # Ultralytics 기본값: 1.6
            min_box_area=tracker_config.get("min_box_area", 100),     # Ultralytics 기본값: 100
            mot20=tracker_config.get("mot20", False),                 # Ultralytics 기본값: False
        )
        self.frame_rate = tracker_config.get("frame_rate", 30)
        self.tracker = BYTETracker(args, frame_rate=self.frame_rate)

    def update_tracks(self, detections, frame, frame_id: int, fps: float = None):
        # BYTETracker가 받는 dets: [[x, y, w, h, score], ...]
        if detections:
            dets = np.array([[d[0], d[1], d[2], d[3], d[4]] for d in detections], dtype=np.float32)
        else:
            dets = np.empty((0, 5), dtype=np.float32)

        img_h, img_w = frame.shape[:2]
        online_targets = self.tracker.update(
            dets,
            [img_h, img_w],      # img_info
            [img_h, img_w]       # img_size
        )

        # 필요 형식으로 반환
        out = []
        for t in online_targets:
            tlwh = t.tlwh
            # tlwh를 bbox 형식으로 변환 (x1, y1, x2, y2)
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            out.append({
                "track_id": int(t.track_id), 
                "bbox": [x1, y1, x2, y2]
            })
        return out
