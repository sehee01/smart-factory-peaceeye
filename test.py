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
    sys.path.append('deep-person-reid-master')
    # sys.path.append('TensorRT-8.5.3.1')
    from yolox.tracker.byte_tracker import BYTETracker, STrack as OriginalSTrack, TrackState
    from yolox.tracker.kalman_filter import KalmanFilter
    from yolox.tracker.matching import iou_distance, linear_assignment
    from torchreid.utils.feature_extractor import FeatureExtractor
    # import tensorrt as trt
except ImportError as e:
    print(f"필수 라이브러리 로드 실패: {e}")
    print("ByteTrack 또는 deep-person-reid 경로를 확인하거나 'pip install -r requirements.txt'를 실행하세요.")
    sys.exit(1)



# --- np.float 호환성 문제 해결 ---
# numpy 1.24.0 이상 버전에서 np.float이 제거됨에 따라 float으로 대체합니다.
if not hasattr(np, 'float'):
    np.float = float

from collections import deque
from datetime import datetime, timedelta

class CrossCameraReIDManager:
    def __init__(self, similarity_threshold=0.7, feature_ttl=300):
        """
        Args:
            similarity_threshold: Re-ID 매칭 임계값
            feature_ttl: 특징 벡터 유효 시간(초)
        """
        self.features_db = {}  # {global_id: {'features': [], 'last_seen': datetime, 'cameras': set()}}
        self.camera_to_global = {}  # {(camera_id, local_track_id): global_id}
        self.lock = threading.Lock()
        self.similarity_threshold = similarity_threshold
        self.feature_ttl = feature_ttl
        self.next_global_id = 1
        
    def match_or_create(self, camera_id, local_track_id, feature_vector, bbox_info):
        """
        새로운 특징 벡터를 기존 DB와 비교하여 매칭하거나 새 ID 생성
        
        Returns:
            global_id: 전역 추적 ID
            is_new: 새로운 ID인지 여부
        """
        with self.lock:
            # 이미 매핑된 경우
            if (camera_id, local_track_id) in self.camera_to_global:
                global_id = self.camera_to_global[(camera_id, local_track_id)]
                self._update_features(global_id, feature_vector, camera_id)
                return global_id, False
            
            # 만료된 특징 벡터 정리
            self._cleanup_expired_features()
            
            # 가장 유사한 기존 ID 찾기
            best_match_id = None
            best_similarity = 0
            
            for global_id, data in self.features_db.items():
                # 같은 카메라에서 이미 추적 중인 ID는 제외
                if camera_id in data['cameras']:
                    continue
                    
                similarity = self._calculate_similarity(feature_vector, data['features'])
                if similarity > self.similarity_threshold and similarity > best_similarity:
                    best_match_id = global_id
                    best_similarity = similarity
            
            if best_match_id:
                # 기존 ID와 매칭
                self.camera_to_global[(camera_id, local_track_id)] = best_match_id
                self._update_features(best_match_id, feature_vector, camera_id)
                return best_match_id, False
            else:
                # 새로운 전역 ID 생성
                new_global_id = self.next_global_id
                self.next_global_id += 1
                
                self.features_db[new_global_id] = {
                    'features': deque([feature_vector], maxlen=10),  # 최근 10개 특징만 유지
                    'last_seen': datetime.now(),
                    'cameras': {camera_id},
                    'first_bbox': bbox_info
                }
                self.camera_to_global[(camera_id, local_track_id)] = new_global_id
                return new_global_id, True
    
    def _update_features(self, global_id, feature_vector, camera_id):
        """특징 벡터 업데이트"""
        self.features_db[global_id]['features'].append(feature_vector)
        self.features_db[global_id]['last_seen'] = datetime.now()
        self.features_db[global_id]['cameras'].add(camera_id)
    
    def _calculate_similarity(self, query_feature, feature_list):
        """코사인 유사도 계산 (평균 특징 벡터 사용)"""
        if not feature_list:
            return 0
        
        # 최근 특징 벡터들의 평균 계산
        avg_feature = np.mean(list(feature_list), axis=0)
        avg_feature /= np.linalg.norm(avg_feature)
        query_feature = query_feature / np.linalg.norm(query_feature)
        
        return np.dot(query_feature, avg_feature)
    
    def _cleanup_expired_features(self):
        """만료된 특징 벡터 제거"""
        current_time = datetime.now()
        expired_ids = []
        
        for global_id, data in self.features_db.items():
            if (current_time - data['last_seen']).seconds > self.feature_ttl:
                expired_ids.append(global_id)
        
        for global_id in expired_ids:
            # 관련 매핑 제거
            keys_to_remove = [(cam, track) for (cam, track), gid in self.camera_to_global.items() if gid == global_id]
            for key in keys_to_remove:
                del self.camera_to_global[key]
            del self.features_db[global_id]
    
    def remove_track(self, camera_id, local_track_id):
        """특정 카메라의 트랙 제거"""
        with self.lock:
            if (camera_id, local_track_id) in self.camera_to_global:
                global_id = self.camera_to_global[(camera_id, local_track_id)]
                del self.camera_to_global[(camera_id, local_track_id)]
                
                # 해당 카메라 제거
                if global_id in self.features_db:
                    self.features_db[global_id]['cameras'].discard(camera_id)
                    
                    # 더 이상 추적하는 카메라가 없으면 일정 시간 후 자동 삭제됨

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

# --- IOU만 사용하는 BYTETracker ---
class BYTETrackerWithReID(BYTETracker):
    """
    IOU 매칭만 사용하는 ByteTrack 클래스 (Re-ID 매칭 제거)
    """
    def __init__(self, args, frame_rate=30):
        super().__init__(args, frame_rate)

    def update(self, dets, img_info, img, reid_extractor):
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # 1. 감지된 객체(dets) 전처리 - ByteTrack 방식
        scores = dets[:, 4]
        bboxes = dets[:, :4]
        
        # 높은 신뢰도 객체 (track_thresh 이상)
        high_conf_inds = scores > self.args.track_thresh
        high_conf_dets = bboxes[high_conf_inds]
        high_conf_scores = scores[high_conf_inds]
        
        # 낮은 신뢰도 객체 (track_thresh 미만, 0.1 이상)
        low_conf_inds = (scores < self.args.track_thresh) & (scores > 0.1)
        low_conf_dets = bboxes[low_conf_inds]
        low_conf_scores = scores[low_conf_inds]
        
        # 높은 신뢰도 객체 처리
        if high_conf_dets.shape[0] > 0:
            # 이미지에서 객체 부분만 잘라내 Re-ID 특징 추출 (글로벌 Re-ID용)
            crops = []
            for box in high_conf_dets:
                x1, y1, x2, y2 = map(int, box)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    crop = np.zeros((128, 64, 3), dtype=np.uint8)
                crops.append(crop)
            
            features = reid_extractor(crops).cpu().numpy()
            high_conf_detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for tlbr, s in zip(high_conf_dets, high_conf_scores)]
            for d, f in zip(high_conf_detections, features):
                d.update_features(f)
        else:
            high_conf_detections = []
        
        # 낮은 신뢰도 객체 처리
        if low_conf_dets.shape[0] > 0:
            low_conf_detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for tlbr, s in zip(low_conf_dets, low_conf_scores)]
        else:
            low_conf_detections = []

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

        # 3. ByteTrack 원래 로직 구현
        # 3-1. 높은 신뢰도 객체로 기존 트랙 매칭
        if len(high_conf_detections) > 0:
            dists = iou_distance(strack_pool, high_conf_detections)
            matches, u_track, u_detection = linear_assignment(dists, thresh=self.args.match_thresh)
            
            print(f"High-conf matching: {len(matches)} matches, {len(u_track)} unmatched tracks, {len(u_detection)} unmatched detections")

            for itracked, idet in matches:
                track = strack_pool[itracked]
                det = high_conf_detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

        # 3-2. 낮은 신뢰도 객체로 잃어버린 트랙 복구
        if len(low_conf_detections) > 0:
            remaining_lost_tracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Lost]
            
            print(f"Low-conf recovery: {len(low_conf_detections)} low-conf detections, {len(remaining_lost_tracks)} remaining lost tracks")
            
            if len(remaining_lost_tracks) > 0:
                dists = iou_distance(remaining_lost_tracks, low_conf_detections)
                matches, u_lost, u_low_det = linear_assignment(dists, thresh=0.5)
                
                print(f"Low-conf matching: {len(matches)} recovered tracks")

                for ilost, idet in matches:
                    track = remaining_lost_tracks[ilost]
                    det = low_conf_detections[idet]
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

        # 3-3. unconfirmed tracks 처리
        if len(unconfirmed) > 0 and len(high_conf_detections) > 0:
            dists = iou_distance(unconfirmed, high_conf_detections)
            matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
            
            print(f"Unconfirmed matching: {len(matches)} matches, {len(u_unconfirmed)} unmatched unconfirmed, {len(u_detection)} unmatched detections")
            
            for itracked, idet in matches:
                track = unconfirmed[itracked]
                det = high_conf_detections[idet]
                track.update(det, self.frame_id)
                activated_stracks.append(track)

        # 4. 나머지 트랙 처리
        remaining_tracked_pool = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        for track in remaining_tracked_pool:
            track.mark_lost()
            lost_stracks.append(track)

        # 5. 새로운 트랙 생성 (매칭되지 않은 높은 신뢰도 객체)
        new_tracks_created = 0
        for i in u_detection:
            track = high_conf_detections[i]
            if track.score >= self.args.track_thresh:
                track.activate(self.kalman_filter, self.frame_id)
                activated_stracks.append(track)
                new_tracks_created += 1
        
        if new_tracks_created > 0:
            print(f"New tracks created: {new_tracks_created}")
        else:
            print(f"No new tracks created from {len(u_detection)} unmatched detections")

        # 7. 최종 트랙 리스트 정리
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        
        # 8. 트랙 정리: track_buffer 프레임 이상 lost 상태인 트랙 제거
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.args.track_buffer:
                track.state = TrackState.Removed
        
        # 제거된 트랙들을 removed_stracks에 추가
        self.removed_stracks = [t for t in self.lost_stracks if t.state == TrackState.Removed]
        self.lost_stracks = [t for t in self.lost_stracks if t.state != TrackState.Removed]
        
        # 중복 트랙 제거
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

def run_tracking(video_path, yolo_model_path, reid_extractor, frame_queue, stop_event, reid_manager):
    """ByteTrack 추적 함수 (IOU 매칭만 사용, 글로벌 Re-ID용 특징 추출 포함)"""
    try:
        print(f"Starting tracking thread for {video_path}")
        model = YOLO(yolo_model_path, task="detect")
        classNames = model.names
        camera_id = cli_args.videos.index(video_path)  # 카메라 ID

        tracker_args = argparse.Namespace(track_thresh=0.1, match_thresh=0.8, track_buffer=30, mot20=False)
        tracker = BYTETrackerWithReID(tracker_args, frame_rate=30)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            frame_queue.put((video_path, None))
            return
            
        local_to_global_map = {}  # 로컬 ID -> 전역 ID 매핑
        frame_count = 0

        while cap.isOpened() and not stop_event.is_set():
            frame_count += 1
            ret, frame = cap.read()
            if not ret:
                print(f"End of video {video_path} reached at frame {frame_count}")
                break

            try:
                # 프레임 처리 (기존 코드와 동일)
                frame_height, frame_width = frame.shape[:2]
                target_width = 640
                scale = target_width / frame_width
                target_height = int(frame_height * scale)
                frame = cv2.resize(frame, (target_width, target_height))

                detection_results = model(frame, verbose=False, half=torch.cuda.is_available())[0]
                dets = []
                for box in detection_results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if classNames[cls_id].lower() in ["person", "persona"]:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        dets.append([x1, y1, x2, y2, conf])

                # 프레임 번호와 탐지 결과 출력
                frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                print(f"[{video_path}] Frame {frame_num}: Detected {len(dets)} persons, Active: {len(tracker.tracked_stracks)}, Lost: {len(tracker.lost_stracks)}, Removed: {len(tracker.removed_stracks)}")
                
                # 신뢰도 정보 출력
                if len(dets) > 0:
                    confidences = [det[4] for det in dets]
                    print(f"[{video_path}] Confidences: {[f'{c:.3f}' for c in confidences]}")
                
                # 모든 프레임에서 트래킹 업데이트 실행 (탐지된 객체가 없어도)
                if len(dets) > 0:
                    online_targets = tracker.update(torch.tensor(dets), frame.shape[:2], frame, reid_extractor)
                else:
                    # 탐지된 객체가 없어도 빈 텐서로 트래킹 업데이트
                    online_targets = tracker.update(torch.empty((0, 5)), frame.shape[:2], frame, reid_extractor)
                
                # 바운딩 박스 그리기 (탐지된 객체가 있든 없든 트래킹 결과 표시)
                frame_detections_json = []
                for t in online_targets:
                    local_id = t.track_id
                    
                    # Cross-camera Re-ID 수행 (글로벌 Re-ID용)
                    bbox_info = {
                        'camera_id': camera_id,
                        'bbox': t.tlbr.tolist(),
                        'timestamp': datetime.now()
                    }
                    
                    global_id, is_new = reid_manager.match_or_create(
                        camera_id, 
                        local_id, 
                        t.smooth_feat,  # 안정화된 특징 벡터 사용
                        bbox_info
                    )
                    
                    # 로컬-전역 ID 매핑 저장
                    local_to_global_map[local_id] = global_id
                    
                    xmin, ymin, xmax, ymax = map(int, t.tlbr)
                    point_x = (xmin + xmax) / 2
                    point_y = ymin
                    
                    detection_data = {
                        "camera_id": camera_id,
                        "local_track_id": int(local_id),
                        "global_track_id": int(global_id),  # 전역 ID 추가
                        "bbox_xyxy": [point_x, point_y],
                        "is_new_global": is_new
                    }
                    frame_detections_json.append(detection_data)

                    # 화면에 전역 ID 표시
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, f'G:{global_id} L:{local_id}', 
                               (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if frame_detections_json:
                    print(f"--- Camera {camera_id} ({video_path}) ---")
                    print(json.dumps(frame_detections_json, indent=2))

                frame_queue.put((video_path, frame))
                
            except Exception as e:
                print(f"Error processing frame {frame_count} in {video_path}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 추적 종료된 ID들 정리
        for local_id in local_to_global_map:
            reid_manager.remove_track(camera_id, local_id)
        
        cap.release()
        print(f"Tracking thread for {video_path} finished normally")
        frame_queue.put((video_path, None))
        
    except Exception as e:
        print(f"Critical error in tracking thread for {video_path}: {e}")
        import traceback
        traceback.print_exc()
        frame_queue.put((video_path, None))

def main(args):
    reid_extractor = FeatureExtractor(
        model_name='osnet_ibn_x1_0',
        model_path=args.reid_model if args.reid_model else None,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Cross-camera Re-ID 매니저 생성
    reid_manager = CrossCameraReIDManager(
        similarity_threshold=0.65,  # 임계값 조정 필요
        feature_ttl=300  # 5분간 특징 벡터 유지
    )
    
    frame_queue = queue.Queue()
    stop_event = threading.Event()
    
    threads = []
    for video_path in args.videos:
        thread = threading.Thread(
            target=run_tracking, 
            args=(video_path, args.yolo_model, reid_extractor, frame_queue, stop_event, reid_manager),
            daemon=True
        )
        threads.append(thread)
        thread.start()

    # 이하 동일...

    latest_frames = {}
    active_videos = set(args.videos)

    print(f"Started {len(threads)} tracking threads")
    
    while active_videos:
        try:
            # 큐에서 프레임 가져오기 (타임아웃을 사용하여 GUI가 멈추지 않도록 함)
            video_path, frame = frame_queue.get(timeout=0.1)

            if frame is None:  # 스레드 종료 신호
                print(f"Thread for {video_path} finished")
                active_videos.discard(video_path)
                if video_path in latest_frames:
                    del latest_frames[video_path]
                cv2.destroyWindow(f"Tracking - {video_path}")
                continue
            
            latest_frames[video_path] = frame

        except queue.Empty:
            # 큐가 비어있으면 현재 프레임들을 다시 그림
            pass
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()

        # 모든 활성 비디오의 최신 프레임을 화면에 표시
        for path, f in latest_frames.items():
            try:
                cv2.imshow(f"Tracking - {path}", f)
            except Exception as e:
                print(f"Error displaying {path}: {e}")

        # 'q'를 누르면 모든 스레드에 종료 신호를 보내고 루프 탈출
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User requested exit")
            stop_event.set()
            break
    
    print("Waiting for threads to finish...")
    # 모든 스레드가 정상적으로 종료될 때까지 대기
    for i, thread in enumerate(threads):
        print(f"Waiting for thread {i+1}/{len(threads)} to finish...")
        thread.join(timeout=5.0)  # 5초 타임아웃
        if thread.is_alive():
            print(f"Warning: Thread {i+1} did not finish within timeout")
    
    print("All threads finished")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv8 with ByteTrack for Multi-Video Tracking (IOU only, Global Re-ID)")
    parser.add_argument('--videos', nargs='+', type=str, default=["test_video/KSEB03.mp4","test_video/KSEB02.mp4"], help='List of video file paths.')
    parser.add_argument('--yolo_model', type=str, default="models/weights/yolo11m.pt", help='Path to the YOLOv11 model file.')
    parser.add_argument('--reid_model', type=str, default="", help='Path to the Re-ID model weights. Leave empty to download pretrained.')
    
    cli_args = parser.parse_args()
    main(cli_args)
