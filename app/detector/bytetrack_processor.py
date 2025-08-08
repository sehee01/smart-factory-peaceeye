from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
import numpy as np


class ByteTrackProcessor:
    """
    BYTETracker 전용 객체로 추적 알고리즘 단독 실행 책임을 담당.
    - 감지된 박스 정보 (dets)를 기반으로 트랙 유지
    - 내부에서 BYTETracker를 캡슐화하여 추적 처리
    """

    def __init__(self, tracker_config: dict):
        """
        :param tracker_config: BYTETracker 설정 딕셔너리
        """
        self.tracker = BYTETracker(tracker_config)

    def update_tracks(self, detections: list, frame, frame_id: int, fps: float = 30.0):
        """
        BYTETracker에 감지 결과를 입력으로 주고 트래킹 결과를 반환

        :param detections: list of [x, y, w, h, score, cls]
        :param frame: OpenCV BGR 이미지
        :param frame_id: 프레임 번호
        :param fps: FPS 값 (기본 30.0)
        :return: List[track] 객체 리스트
        """
        dets_np = np.array(detections, dtype=np.float32)
        track_results = self.tracker.update(dets_np, frame, frame_id, fps)
        return track_results

    def reset(self):
        """
        트래커 상태 초기화
        """
        self.tracker.reset()
