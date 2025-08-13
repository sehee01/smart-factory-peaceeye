import cv2
import json
import torch
import sys
import threading
import numpy as np
import argparse
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from collections import defaultdict
from models.mapping.homography_calibration2 import HomographyCalibrator
import os

# 경로 설정 및 필수 모듈 로드
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, 'ByteTrack'))  # yolox 포함
    sys.path.append(os.path.join(BASE_DIR, 'ByteTrack', 'yolox'))  # yolox 내부
    sys.path.append(os.path.join(BASE_DIR, 'models', 'mapping'))
    sys.path.append(os.path.join(BASE_DIR, 'deep-person-reid-master'))
    from yolox.tracker.byte_tracker import BYTETracker, STrack as OriginalSTrack, TrackState
    from yolox.tracker.kalman_filter import KalmanFilter
    from yolox.tracker.matching import iou_distance, linear_assignment
    from torchreid.utils.feature_extractor import FeatureExtractor
except ImportError as e:
    print(f"[Error] 필수 라이브러리 로드 실패: {e}")
    sys.exit(1)

if not hasattr(np, 'float'):
    np.float = float

class STrack(OriginalSTrack):
    def __init__(self, tlwh, score, *args, **kwargs):
        super().__init__(tlwh, score, *args, **kwargs)
        self.smooth_feat = None
        self.curr_feat = None
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    @staticmethod
    def multi_predict(stracks, kalman_filter):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance.copy() for st in stracks])
            multi_mean, multi_covariance = kalman_filter.multi_predict(multi_mean, multi_covariance)
            for i, st in enumerate(stracks):
                st.mean = multi_mean[i]
                st.covariance = multi_covariance[i]

class BYTETrackerWithReID(BYTETracker):
    def __init__(self, args, frame_rate=30):
        super().__init__(args, frame_rate)
        self.reid_thresh = 0.6

    def update(self, dets, img_info, img, reid_extractor):
        self.frame_id += 1
        activated_stracks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []

        if dets.shape[0] == 0:
            self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
            return self.tracked_stracks

        scores = dets[:, 4]
        bboxes = dets[:, :4]

        remain_inds = scores > self.args.track_thresh
        high_conf_dets = bboxes[remain_inds]
        high_conf_scores = scores[remain_inds]

        detections = []
        if len(high_conf_dets) > 0:
            crops = []
            for box in high_conf_dets:
                x1, y1, x2, y2 = map(int, box)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    crop = np.zeros((128, 64, 3), dtype=np.uint8)
                crops.append(crop)

            features = reid_extractor(crops).cpu().numpy()
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for tlbr, s in zip(high_conf_dets, high_conf_scores)]
            for det, feat in zip(detections, features):
                det.update_features(feat)

        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool, self.kalman_filter)

        dists = iou_distance(strack_pool, detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        lost_pool = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Lost]
        remaining_detections = [detections[i] for i in u_detection]

        if lost_pool and remaining_detections:
            lost_features = np.array([track.smooth_feat for track in lost_pool])
            det_features = np.array([det.smooth_feat for det in remaining_detections])
            feature_dists = cdist(lost_features, det_features, 'cosine')
            reid_matches, u_lost, u_det_reid = linear_assignment(feature_dists, thresh=self.reid_thresh)

            for ilost, idet in reid_matches:
                track = lost_pool[ilost]
                det = remaining_detections[idet]
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        remaining_tracked_pool = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        for track in remaining_tracked_pool:
            track.mark_lost()
            lost_stracks.append(track)

        for i in u_detection:
            track = detections[i]
            if track.score >= self.args.track_thresh:
                track.activate(self.kalman_filter, self.frame_id)
                activated_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(self.lost_stracks)

        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        return [t for t in self.tracked_stracks if t.is_activated]

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

def load_homography_matrix_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    matrix_lines = [line.strip() for line in lines if "[" in line]
    matrix = [list(map(float, line.replace('[', '').replace(']', '').split(','))) for line in matrix_lines]
    return np.array(matrix)

MAP_CANVAS_WIDTH, MAP_CANVAS_HEIGHT = 800, 600
map_canvas = np.ones((MAP_CANVAS_HEIGHT, MAP_CANVAS_WIDTH, 3), dtype=np.uint8) * 255
all_track_histories = defaultdict(lambda: defaultdict(list))
camera_colors = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0), 3: (255, 255, 0)}

def run_tracking_with_mapping(video_path, yolo_model_path, reid_extractor, camera_id):
    model = YOLO(yolo_model_path)
    class_names = model.names
    tracker_args = argparse.Namespace(track_thresh=0.5, match_thresh=0.8, track_buffer=30, mot20=False)
    tracker = BYTETrackerWithReID(tracker_args)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] 비디오를 열 수 없습니다: {video_path}")
        return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"카메라 {camera_id} 영상 크기: {frame_width}x{frame_height}")


    homography_path = rf"C:\Users\user\Desktop\smartfactoryPeaceeye\smart-factory-peaceeye\result\matrix\homography_matrix_{camera_id}.txt"
    if not os.path.exists(homography_path):
        print(f"[Error] Homography 행렬 파일이 없습니다: {homography_path}")
        return
    homography_matrix = load_homography_matrix_from_txt(homography_path)
    calibrator = HomographyCalibrator(homography_matrix=homography_matrix)

    scale = 640 / frame_width
    print(f"스케일: {scale}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_frame = cv2.resize(frame, (640, int(frame_height * scale)))
        results = model(input_frame)[0]

        dets = []
        for result in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls_id = result
            cls_id = int(cls_id)
            if class_names[cls_id] != 'person' or conf < 0.3:
                continue
            x1 /= scale
            y1 /= scale
            x2 /= scale
            y2 /= scale
            dets.append([x1, y1, x2, y2, conf])
        dets = np.array(dets)

        tracked_stracks = tracker.update(dets, (frame_width, frame_height), frame, reid_extractor)

        for track in tracked_stracks:
            if not track.is_activated or track.state != TrackState.Tracked:
                continue
            tlwh = track.tlwh
            x_center = tlwh[0] + tlwh[2] / 2
            y_center = tlwh[1] + tlwh[3]
            map_x, map_y = calibrator.pixel_to_map_coordinates((x_center, y_center))
            track_id = track.track_id

            all_track_histories[camera_id][track_id].append((int(map_x), int(map_y)))
            color = camera_colors.get(camera_id, (0, 255, 255))
            points = all_track_histories[camera_id][track_id]
            for i in range(1, len(points)):
                cv2.line(map_canvas, points[i-1], points[i], color, 2)
            if points:
                cv2.circle(map_canvas, points[-1], 5, color, -1)

            x1, y1, w, h = tlwh
            x2, y2 = int(x1 + w), int(y1 + h)
            x1, y1 = int(x1), int(y1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'ID:{track_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow(f'Camera {camera_id}', frame)
        cv2.imshow('2D Map', map_canvas)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    reid_extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='deep-person-reid/model_weights/osnet_x1_0_msmt17.pth',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    video_paths = [
        r'C:\Users\user\Desktop\smartfactoryPeaceeye\smart-factory-peaceeye\test_video\0_te3.mp4',
        r'C:\Users\user\Desktop\smartfactoryPeaceeye\smart-factory-peaceeye\test_video\test01.mp4',
    ]

    yolo_model_path = 'yolov8n.pt'

    threads = []
    for cam_id, video_path in enumerate(video_paths):
        t = threading.Thread(target=run_tracking_with_mapping, args=(video_path, yolo_model_path, reid_extractor, cam_id))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

if __name__ == '__main__':
    main()
