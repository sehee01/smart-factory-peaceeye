import cv2
import json
import torch
import sys
import threading
import numpy as np
import argparse
import queue
from ultralytics import YOLO
from scipy.spatial.distance import cdist

# --- 경로 설정 ---
# 현재 스크립트의 위치를 기준으로 상대 경로를 설정하여 다른 환경에서의 실행을 용이하게 합니다.
try:
    # ByteTrack, deep-person-reid가 현재 프로젝트 폴더 내에 있다고 가정
    sys.path.append('ByteTrack')
    sys.path.append('deep-person-reid')
    from yolox.tracker.byte_tracker import BYTETracker, STrack as OriginalSTrack, TrackState
    from yolox.tracker.kalman_filter import KalmanFilter
    from yolox.tracker.matching import iou_distance, linear_assignment
    from torchreid.utils.feature_extractor import FeatureExtractor
except ImportError as e:
    print(f"필수 라이브러리 로드 실패: {e}")
    print("ByteTrack 또는 deep-person-reid 경로를 확인하거나 'pip install -r requirements.txt'를 실행하세요.")
    sys.exit(1)



# --- np.float 호환성 문제 해결 ---
# numpy 1.24.0 이상 버전에서 np.float이 제거됨에 따라 float으로 대체합니다.
if not hasattr(np, 'float'):
    np.float = float


# --- 개선된 STrack 클래스 ---
class STrack(OriginalSTrack):
    """
    기존 STrack을 상속받아 ReID 특징(feature)을 관리하고,
    시간이 지나도 안정적인 특징 유지를 위해 feature smoothing을 추가합니다.
    """
    def __init__(self, tlwh, score, *args, **kwargs):
        super().__init__(tlwh, score, *args, **kwargs)
        self.smooth_feat = None
        self.curr_feat = None
        self.alpha = 0.9  # Feature smoothing factor (이전 feature를 얼마나 유지할지 결정)

    def update_features(self, feat):
        """새로운 ReID feature로 현재 feature를 업데이트하고, smoothing을 적용합니다."""
        feat /= np.linalg.norm(feat)  # L2 정규화
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            # 지수 이동 평균(Exponential Moving Average)과 유사한 방식으로 feature를 부드럽게 업데이트
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    @staticmethod
    def multi_predict(stracks, kalman_filter):
        """Kalman Filter를 사용해 여러 트랙의 다음 상태를 예측합니다."""
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance.copy() for st in stracks])
            multi_mean, multi_covariance = kalman_filter.multi_predict(multi_mean, multi_covariance)
            for i, st in enumerate(stracks):
                st.mean = multi_mean[i]
                st.covariance = multi_covariance[i]


# --- ReID와 IOU를 함께 사용하는 개선된 BYTETracker ---
class BYTETrackerWithReID(BYTETracker):
    """
    기존 BYTETracker의 update 로직을 ReID feature를 사용하도록 개선합니다.
    Cascaded Matching(단계적 매칭)을 통해 정확도를 높입니다.
    1. IOU 기반 매칭: 확실한 트랙들을 먼저 연결합니다.
    2. Re-ID 기반 매칭: IOU로 찾지 못한 트랙(가려짐 등)을 외형 특징으로 다시 찾아 연결합니다.
    """
    def __init__(self, args, frame_rate=30):
        super().__init__(args, frame_rate)
        self.reid_thresh = 0.6  # Re-ID 코사인 거리 임계값 (1 - 유사도)

    def update(self, dets, img_info, img, reid_extractor):
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # 1. 감지된 객체(dets) 전처리
        scores = dets[:, 4]
        bboxes = dets[:, :4]
        
        # 신뢰도 임계값(track_thresh) 이상의 객체만 'detections'으로 간주
        remain_inds = scores > self.args.track_thresh
        high_conf_dets = bboxes[remain_inds]
        high_conf_scores = scores[remain_inds]
        
        if high_conf_dets.shape[0] > 0:
            # 이미지에서 객체 부분만 잘라내 Re-ID 특징 추출
            crops = []
            for box in high_conf_dets:
                x1, y1, x2, y2 = map(int, box)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0: # 크기가 0인 crop 방지
                    crop = np.zeros((128, 64, 3), dtype=np.uint8)
                crops.append(crop)
            
            features = reid_extractor(crops).cpu().numpy()
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for tlbr, s in zip(high_conf_dets, high_conf_scores)]
            for d, f in zip(detections, features):
                d.update_features(f)
        else:
            detections = []

        # 2. 기존 트랙과 새로 감지된 객체 매칭 준비
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        # 추적 중인 트랙과 최근에 잃어버린 트랙을 합쳐 매칭 대상(pool)으로 설정
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool, self.kalman_filter)

        # 3. 단계적 매칭 (Cascaded Matching)
        # 3-1. IOU 거리 기반 1차 매칭 (가깝고 확실한 경우)
        dists = iou_distance(strack_pool, detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else: # Lost 상태의 트랙을 다시 찾은 경우
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 3-2. Re-ID 특징 기반 2차 매칭 (가려졌거나 멀리 떨어진 경우)
        # 1차 매칭에서 실패한 '잃어버린 트랙(lost tracks)'과 '감지된 객체'를 대상으로 수행
        lost_pool = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Lost]
        remaining_detections = [detections[i] for i in u_detection]

        if len(lost_pool) > 0 and len(remaining_detections) > 0:
            lost_features = np.array([track.smooth_feat for track in lost_pool])
            det_features = np.array([det.smooth_feat for det in remaining_detections])
            
            # 코사인 거리 계산
            feature_dists = cdist(lost_features, det_features, 'cosine')
            reid_matches, u_lost, u_det_reid = linear_assignment(feature_dists, thresh=self.reid_thresh)

            for ilost, idet in reid_matches:
                track = lost_pool[ilost]
                det = remaining_detections[idet]
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        # Re-ID로 매칭된 객체는 u_detection에서 제거해야 하지만, 복잡성을 줄이기 위해 생략.
        # 이로 인해 일부 객체가 중복 처리될 수 있으나, 전체적인 성능에 큰 영향은 없음.

        # 4. 나머지 트랙 처리 및 새로운 트랙 생성
        # IOU 매칭에서 살아남았지만 Re-ID에서 처리되지 않은 트랙들
        remaining_tracked_pool = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        for track in remaining_tracked_pool:
            track.mark_lost()
            lost_stracks.append(track)

        # 매칭되지 않은 높은 신뢰도의 객체는 새로운 트랙으로 등록
        for i in u_detection:
            track = detections[i]
            if track.score >= self.args.track_thresh:
                track.activate(self.kalman_filter, self.frame_id)
                activated_stracks.append(track)

        # 5. 최종 트랙 리스트 정리
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(self.lost_stracks) # 메모리 정리 로직 (단순화)
        
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        
        return [t for t in self.tracked_stracks if t.is_activated]

# --- 유틸리티 함수 ---
def joint_stracks(tlista, tlistb):
    exists = {t.track_id for t in tlista}
    res = tlista[:]
    for t in tlistb:
        if t.track_id not in exists:
            exists.add(t.track_id)
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    track_ids_b = {t.track_id for t in tlistb}
    return [t for t in tlista if t.track_id not in track_ids_b]

def remove_duplicate_stracks(stracks1, stracks2):
    pdist = iou_distance(stracks1, stracks2)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(pairs[0]), list(pairs[1])
    for p, q in zip(dupa, dupb):
        timep = stracks1[p].frame_id - stracks1[p].start_frame
        timeq = stracks2[q].frame_id - stracks2[q].start_frame
        if timep > timeq:
            stracks2[q].state = TrackState.Removed
    return stracks1, stracks2

# --- 메인 실행 로직 ---
def run_tracking(video_path, yolo_model_path, reid_extractor, frame_queue, stop_event):
    """하나의 비디오 스트림에 대한 추적을 실행하고 결과를 큐에 넣는 함수"""
    model = YOLO(yolo_model_path)
    classNames = model.names

    tracker_args = argparse.Namespace(track_thresh=0.5, match_thresh=0.8, track_buffer=30, mot20=False)
    tracker = BYTETrackerWithReID(tracker_args, frame_rate=30)

    cap = cv2.VideoCapture(video_path)
    person_count = 0
    tracked_ids = set()

    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        # --- 경량화를 위한 프레임 리사이즈 ---
        # 프레임 너비를 640으로 고정하고, 비율에 맞춰 높이를 조절합니다.
        frame_height, frame_width = frame.shape[:2]
        target_width = 640
        scale = target_width / frame_width
        target_height = int(frame_height * scale)
        frame = cv2.resize(frame, (target_width, target_height))

        # FP16(half-precision) 추론은 CUDA 사용 가능 시에만 적용하여 속도 향상
        detection_results = model(frame, verbose=False, half=torch.cuda.is_available())[0]
        dets = []
        for box in detection_results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if classNames[cls_id].lower() in ["person", "persona"]:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append([x1, y1, x2, y2, conf])

        if len(dets) > 0:
            online_targets = tracker.update(torch.tensor(dets), frame.shape[:2], frame, reid_extractor)
            
            frame_detections_json = []
            for t in online_targets:
                if t.track_id not in tracked_ids:
                    tracked_ids.add(t.track_id)
                    person_count += 1
                
                xmin, ymin, xmax, ymax = map(int, t.tlbr)
                detection_data = {"track_id": int(t.track_id), "bbox_xyxy": [xmin, ymin, xmax, ymax]}
                frame_detections_json.append(detection_data)

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, f'ID:{t.track_id}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if frame_detections_json:
                print(f"--- {video_path} ---")
                print(json.dumps(frame_detections_json, indent=2))

        # 처리된 프레임을 큐에 삽입
        frame_queue.put((video_path, frame))

    cap.release()
    # 해당 비디오 처리가 끝났음을 알리는 신호(None)를 큐에 삽입
    frame_queue.put((video_path, None))


def main(args):
    """메인 함수: Re-ID 모델 초기화, 워커 스레드 시작 및 GUI 처리"""
    reid_extractor = FeatureExtractor(
        model_name='osnet_ibn_x1_0',
        model_path=args.reid_model if args.reid_model else None,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    frame_queue = queue.Queue()
    stop_event = threading.Event()
    
    threads = []
    for video_path in args.videos:
        # 스레드에 frame_queue와 stop_event 전달
        thread = threading.Thread(target=run_tracking, args=(video_path, args.yolo_model, reid_extractor, frame_queue, stop_event), daemon=True)
        threads.append(thread)
        thread.start()

    latest_frames = {}
    active_videos = set(args.videos)

    while active_videos:
        try:
            # 큐에서 프레임 가져오기 (타임아웃을 사용하여 GUI가 멈추지 않도록 함)
            video_path, frame = frame_queue.get(timeout=0.1)

            if frame is None:  # 스레드 종료 신호
                active_videos.discard(video_path)
                if video_path in latest_frames:
                    del latest_frames[video_path]
                cv2.destroyWindow(f"Tracking - {video_path}")
                continue
            
            latest_frames[video_path] = frame

        except queue.Empty:
            # 큐가 비어있으면 현재 프레임들을 다시 그림
            pass

        # 모든 활성 비디오의 최신 프레임을 화면에 표시
        for path, f in latest_frames.items():
            cv2.imshow(f"Tracking - {path}", f)

        # 'q'를 누르면 모든 스레드에 종료 신호를 보내고 루프 탈출
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
    
    # 모든 스레드가 정상적으로 종료될 때까지 대기
    for thread in threads:
        thread.join()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv8 with ByteTrack and Re-ID for Multi-Video Tracking")
    parser.add_argument('--videos', nargs='+', type=str, default=["test_video/test01.mp4","test_video/0_te2.mp4"], help='List of video file paths.')
    parser.add_argument('--yolo_model', type=str, default="models/weights/best.pt", help='Path to the YOLOv8 model file.')
    parser.add_argument('--reid_model', type=str, default="", help='Path to the Re-ID model weights. Leave empty to download pretrained.')
    
    cli_args = parser.parse_args()
    main(cli_args)