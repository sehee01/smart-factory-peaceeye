import cv2
import time
import numpy as np
from detector.detector_manager import ByteTrackDetectorManager
from reid.reid_manager import GlobalReIDManager
from reid.redis_handler import FeatureStoreRedisHandler
from reid.similarity import FeatureSimilarityCalculator


class AppOrchestrator:
    """
    전체 객체 탐지, 추적, ReID 재부여 흐름을 통합 실행하는 오케스트레이터
    """

    def __init__(self, model_path: str, tracker_config: dict, redis_conf: dict, reid_conf: dict):
        self.detector = ByteTrackDetectorManager(model_path, tracker_config)

        redis_handler = FeatureStoreRedisHandler(
            redis_host=redis_conf.get("host", "localhost"),
            redis_port=redis_conf.get("port", 6379),
            feature_ttl=reid_conf.get("ttl", 300)
        )
        similarity = FeatureSimilarityCalculator()
        self.reid = GlobalReIDManager(redis_handler, similarity, similarity_threshold=reid_conf.get("threshold", 0.7))

        self.camera_id = redis_conf.get("camera_id", "cam01")

    def run_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            track_list = self.detector.detect_and_track(frame, frame_id)

            for track in track_list:
                local_id = track["track_id"]
                bbox = track["bbox"]

                # --- Feature 추출 로직 (예시: crop 후 평균 RGB 벡터로 대체) ---
                x1, y1, x2, y2 = map(int, bbox)
                crop = frame[y1:y2, x1:x2]
                feature = self._extract_feature(crop)

                global_id = self.reid.reassign_global_id(
                    camera_id=self.camera_id,
                    local_track_id=local_id,
                    feature=feature
                )

                # --- 결과 디스플레이 ---
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {global_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _extract_feature(self, crop_img):
        # 임시 feature: 평균 RGB값을 1D 벡터로 변환
        if crop_img.size == 0:
            return np.zeros(3)
        feature = crop_img.mean(axis=(0, 1))  # RGB 평균
        return feature / 255.0


if __name__ == '__main__':
    import argparse
    from config import settings

    parser = argparse.ArgumentParser(
        description="YOLOv8 with ByteTrack and Redis Global Re-ID V2 for Multi-Video Tracking"
    )
    parser.add_argument(
        '--videos',
        nargs='+',
        type=str,
        default=["test_video/KSEB02.mp4", "test_video/KSEB03.mp4"],
        help='List of video file paths.'
    )
    parser.add_argument(
        '--yolo_model',
        type=str,
        default="../models/weights/bestcctv.pt",
        help='Path to the YOLOv8 model file.'
    )
    parser.add_argument(
        '--redis_host',
        type=str,
        default="localhost",
        help='Redis server host.'
    )
    parser.add_argument(
        '--redis_port',
        type=int,
        default=6379,
        help='Redis server port.'
    )
    parser.add_argument(
        '--camera_id',
        type=str,
        default="cam01",
        help='Camera ID (used as source identifier)'
    )
    args = parser.parse_args()

    tracker_config = settings.TRACKER_CONFIG
    reid_config = settings.REID_CONFIG

    for video_path in args.videos:
        print(f"▶ Processing video: {video_path}")
        redis_conf = {
            "host": args.redis_host,
            "port": args.redis_port,
            "camera_id": args.camera_id
        }

        app = AppOrchestrator(
            model_path=args.yolo_model,
            tracker_config=tracker_config,
            redis_conf=redis_conf,
            reid_conf=reid_config
        )
        app.run_video(video_path)
