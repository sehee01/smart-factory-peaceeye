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
# from torchreid.utils.feature_extractor import FeatureExtractor
# 바이트트랙 +리아이디 모두 안정적인 버전 (같은 카메라 내 코드 공유문제 해결 필요요)
# --- 경로 설정 ---
# 현재 스크립트의 위치를 기준으로 상대 경로를 설정하여 다른 환경에서의 실행을 용이하게 합니다.
try:
    # ByteTrack, deep-person-reid가 현재 프로젝트 폴더 내에 있다고 가정
    sys.path.append('ByteTrack')
    sys.path.append('deep-person-reid-master')
    # sys.path.append('TensorRT-8.5.3.1')
    from yolox.tracker.byte_tracker import BYTETracker, STrack, TrackState
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


# --- 글로벌 ReID를 사용하는 BYTETracker ---
class BYTETrackerWithReID(BYTETracker):
    """
    글로벌 ReID를 사용하여 여러 카메라에서 동일한 객체를 식별하는 ByteTrack
    """
    def __init__(self, args, frame_rate=30, camera_id=0):
        super().__init__(args, frame_rate)
        self.camera_id = camera_id
        self.reid_thresh = 0.6  # Re-ID 코사인 거리 임계값
        self.local_id_mapping = {}  # {local_track_id: local_id}

    def update(self, dets, img_info, img_size, reid_extractor, local_reid_manager, frame_id):
        """로컬 ReID를 사용한 업데이트"""
        # test.py와 동일한 방식으로 ByteTrack 업데이트
        online_targets = super().update(dets, img_info, img_size)
        
        # ReID 처리는 run_tracking에서 직접 수행
        return online_targets


# --- 메인 실행 로직 ---
def run_tracking(video_path, yolo_model_path, reid_extractor, frame_queue, stop_event, camera_id=0, local_reid_manager=None):
    """하나의 비디오 스트림에 대한 추적을 실행하고 결과를 큐에 넣는 함수"""
    model = YOLO(yolo_model_path, task="detect")
    classNames = model.names

    tracker_args = argparse.Namespace(track_thresh=0.5, match_thresh=0.8, track_buffer=150, mot20=False)
    tracker = BYTETrackerWithReID(tracker_args, frame_rate=30, camera_id=camera_id)

    cap = cv2.VideoCapture(video_path)
    person_count = 0
    tracked_ids = set()
    frame_id = 0

    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print(f"[{video_path}] Video ended, restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 비디오를 처음으로 되돌림
            continue

        frame_id += 1
        
        # 로컬 ReID 매니저 프레임 업데이트
        if local_reid_manager is not None:
            local_reid_manager.update_frame(frame_id)

        # --- 경량화를 위한 프레임 리사이즈 ---
        # 프레임 너비를 640으로 고정하고, 비율에 맞춰 높이를 조절합니다.
        original_height, original_width = frame.shape[:2]
        print(f"[{video_path}] Original frame size: {original_width}x{original_height}")
        
        target_width = 640
        scale = target_width / original_width
        target_height = int(original_height * scale)
        frame = cv2.resize(frame, (target_width, target_height))
        print(f"[{video_path}] Resized frame size: {target_width}x{target_height}, scale={scale:.3f}")
        
        # FP16(half-precision) 추론은 CUDA 사용 가능 시에만 적용하여 속도 향상
        detection_results = model(frame, verbose=False, half=torch.cuda.is_available())[0]
        dets = []
        for box in detection_results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if classNames[cls_id].lower() in ["person", "saram"]:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append([x1, y1, x2, y2, conf])
                print(f"[{video_path}] Detection in resized frame: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
        
        # 디버깅을 위한 탐지 정보 출력
        print(f"[{video_path}] Detected {len(dets)} persons with confidences: {[f'{d[4]:.3f}' for d in dets]}")

        if len(dets) > 0:
            # numpy 배열을 그대로 사용 (GPU-CPU 이동 없음)
            dets_array = np.array(dets, dtype=np.float32)
            print(f"[{video_path}] Input to ByteTrack (resized coords): {dets_array}")
        else:
            # 빈 numpy 배열
            dets_array = np.empty((0, 5), dtype=np.float32)

        # ByteTrack 내부 좌표 변환 추적
        img_h, img_w = frame.shape[:2]  # (height, width)
        img_size_w, img_size_h = frame.shape[1], frame.shape[0]  # (width, height)
        scale = min(img_size_w / float(img_h), img_size_h / float(img_w))
        print(f"[{video_path}] ByteTrack internal: img_h={img_h}, img_w={img_w}, img_size=({img_size_w},{img_size_h}), scale={scale:.3f}")
        
        # 문제: ByteTrack이 원본 크기로 변환하려고 함
        # 해결: test.py와 동일하게 frame.shape[:2] 전달
        online_targets = tracker.update(torch.tensor(dets_array), frame.shape[:2], frame.shape[:2], reid_extractor, local_reid_manager, frame_id)
        
        # ReID 처리를 run_tracking에서 직접 수행
        if len(online_targets) > 0 and reid_extractor is not None and local_reid_manager is not None:
            # 이미지에서 객체 부분만 잘라내 ReID 특징 추출
            crops = []
            bboxes = []
            for track in online_targets:
                x1, y1, x2, y2 = map(int, track.tlbr)
                crop = frame[y1:y2, x1:x2]  # frame은 실제 이미지
                if crop.size == 0:
                    crop = np.zeros((128, 64, 3), dtype=np.uint8)
                crops.append(crop)
                bboxes.append([x1, y1, x2, y2])
            
            if len(crops) > 0:
                # ReID 특징 추출
                features = reid_extractor(crops).cpu().numpy()
                
                # 매칭된 트랙들을 추적하는 집합
                matched_tracks = set()
                
                # 각 트랙에 대해 로컬 ID 매칭 (중복 매칭 방지)
                for i, track in enumerate(online_targets):
                    if i < len(features):
                        local_id = local_reid_manager.match_or_create(
                            features[i], bboxes[i], camera_id, frame_id, matched_tracks
                        )
                        if local_id is not None:
                            tracker.local_id_mapping[track.track_id] = local_id
                            print(f"Camera {camera_id}: Local ID {track.track_id} → Local ID {local_id}")
        
        # 추적 디버깅 정보 추가
        print(f"[{video_path}] Tracking Debug: dets={len(dets)}, online_targets={len(online_targets)}")
        if len(dets) > 0:
            print(f"[{video_path}] Detection boxes: {dets}")
        
        frame_detections_json = []
        print(f"[{video_path}] Online targets: {len(online_targets)}")
        for t in online_targets:
            if t.track_id not in tracked_ids:
                tracked_ids.add(t.track_id)
                person_count += 1
                print(f"[{video_path}] NEW TRACK ID: {t.track_id} (Total unique IDs: {len(tracked_ids)})")
            
            xmin, ymin, xmax, ymax = map(int, t.tlbr)
            print(f"[{video_path}] Track {t.track_id}: bbox=({xmin},{ymin},{xmax},{ymax})")
            
            # 좌표계 확인: 현재 좌표가 리사이즈된 프레임 기준인지 확인
            print(f"[{video_path}] Track {t.track_id}: bbox in resized frame ({target_width}x{target_height})")
            
            # 좌표 범위 확인
            print(f"[{video_path}] Track {t.track_id}: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
            print(f"[{video_path}] Track {t.track_id}: frame shape = {frame.shape}")
            
            # 좌표가 프레임 범위를 벗어나는지 확인
            if xmin < 0 or ymin < 0 or xmax > frame.shape[1] or ymax > frame.shape[0]:
                print(f"[{video_path}] WARNING: Track {t.track_id} coordinates out of frame bounds!")
                # 좌표를 프레임 범위로 제한
                xmin = max(0, min(xmin, frame.shape[1] - 1))
                ymin = max(0, min(ymin, frame.shape[0] - 1))
                xmax = max(xmin + 1, min(xmax, frame.shape[1]))
                ymax = max(ymin + 1, min(ymax, frame.shape[0]))
                print(f"[{video_path}] Track {t.track_id}: Clipped bbox=({xmin},{ymin},{xmax},{ymax})")
            
            point_x = (xmin+xmax)/2
            point_y = ymin
            
            # 로컬 ID 가져오기
            local_id = tracker.local_id_mapping.get(t.track_id, t.track_id)
            
            detection_data = {"camera_id": int(cli_args.videos.index(video_path)),
                                 "track_id": int(local_id),  # 로컬 ID 사용
                                 "bbox_xyxy": [point_x, point_y],
                                 "has_reid_feature": t.smooth_feat is not None if hasattr(t, 'smooth_feat') else False}
            #####전달해주는 JSON파일
            #"bbox_xyxy": [xmin, ymin, xmax, ymax]
            frame_detections_json.append(detection_data)

            # 그리기 전 좌표 확인
            print(f"[{video_path}] Drawing box for Track {t.track_id} (Local: {local_id}): ({xmin},{ymin}) to ({xmax},{ymax})")
            
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f'ID:{local_id}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if frame_detections_json:
            print(f"--- {video_path} ---")
            print(json.dumps(frame_detections_json, indent=2))
        else:
            print(f"--- {video_path} --- No detections")

        # 처리된 프레임을 큐에 삽입
        frame_queue.put((video_path, frame))

    cap.release()
    # 해당 비디오 처리가 끝났음을 알리는 신호(None)를 큐에 삽입
    frame_queue.put((video_path, None))


# --- 로컬컬 ReID 매니저 ---
class LocalReIDManager:
    """
    여러 카메라에서 동일한 객체를 식별하고 ByteTrack ID를 로컬 ID로 관리하는 클래스
    """
    def __init__(self, similarity_threshold=0.7, feature_ttl=300, max_features=10):
        self.local_tracks = {}  # {local_id: {'features': [], 'last_seen': frame_id, 'camera_id': int}}
        self.similarity_threshold = similarity_threshold
        self.feature_ttl = feature_ttl  # 프레임 단위 TTL
        self.max_features = max_features  # 슬라이딩 윈도우 크기
        self.current_frame = 0
    
    def update_frame(self, frame_id):
        """현재 프레임 업데이트 및 만료된 트랙 정리"""
        self.current_frame = frame_id
        
        # TTL이 만료된 트랙 제거
        expired_tracks = []
        for local_id, track_info in self.local_tracks.items():
            if self.current_frame - track_info['last_seen'] > self.feature_ttl:
                expired_tracks.append(local_id)
        
        for local_id in expired_tracks:
            del self.local_tracks[local_id]
            print(f"Local ReID: Expired track {local_id}")
    
    def match_or_create(self, features, bbox, camera_id, frame_id, matched_tracks=None):
        """
        ReID 특징과 위치 정보를 기반으로 기존 트랙과 매칭하거나 새로운 로컬 ID 생성
        
        Args:
            features: ReID 특징 벡터 (numpy array)
            bbox: 바운딩 박스 [x1, y1, x2, y2]
            camera_id: 카메라 ID
            frame_id: 현재 프레임 ID
            matched_tracks: 이미 매칭된 트랙 ID들의 집합 (중복 매칭 방지용)
            
        Returns:
            local_id: 매칭된 또는 새로 생성된 로컬 ID
        """
        if matched_tracks is None:
            matched_tracks = set()
        if features is None or len(features) == 0: #특징 벡터 유무무
            return None
        
        # 특징 벡터 정규화
        features = features / np.linalg.norm(features)
        
        best_match_id = None # 매칭된 로컬 ID
        best_similarity = 0 # 유사도
        
        # 기존 트랙들과 유사도 계산
        for local_id, track_info in self.local_tracks.items():
            # 이미 매칭된 트랙은 건너뛰기
            if local_id in matched_tracks:
                continue
                
            if len(track_info['features']) == 0:
                continue
            
            # 위치 기반 필터링 (같은 카메라에서만)
            if track_info['camera_id'] == camera_id:
                # 마지막 위치 정보가 있으면 거리 계산
                if 'last_bbox' in track_info:
                    last_bbox = track_info['last_bbox']
                    # 바운딩 박스 중심점 거리 계산
                    current_center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                    last_center = [(last_bbox[0] + last_bbox[2]) / 2, (last_bbox[1] + last_bbox[3]) / 2]
                    distance = np.sqrt((current_center[0] - last_center[0])**2 + (current_center[1] - last_center[1])**2)
                    
                    # 거리가 너무 멀면 건너뛰기 (위치 기반 필터링)
                    max_distance = 200  # 픽셀 단위, 조정 가능
                    if distance > max_distance:
                        continue
            
            # 가중 평균 특징 계산 (최신 특징에 더 높은 가중치)
            features_array = np.array(track_info['features'])
            if len(features_array) == 1:
                # 특징이 하나뿐인 경우
                weighted_average = features_array[0]
            else:
                # 가중치 계산: 최신 특징에 더 높은 가중치 (0.5 ~ 1.0)
                # 예: 3개 특징이 있으면 [0.5, 0.75, 1.0] -> 정규화 후 [0.22, 0.33, 0.45]
                weights = np.linspace(0.5, 1.0, len(features_array))
                weights = weights / np.sum(weights)  # 정규화
                weighted_average = np.average(features_array, axis=0, weights=weights)
            
            similarity = 1 - cdist([features], [weighted_average], 'cosine')[0][0]
            
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_match_id = local_id
        
        if best_match_id is not None:
            # 기존 트랙과 매칭
            self.local_tracks[best_match_id]['features'].append(features)
            self.local_tracks[best_match_id]['last_seen'] = frame_id
            self.local_tracks[best_match_id]['camera_id'] = camera_id
            self.local_tracks[best_match_id]['last_bbox'] = bbox  # 위치 정보 저장
            
            # 매칭된 트랙을 추적 집합에 추가 (중복 매칭 방지)
            matched_tracks.add(best_match_id)
            
            # 슬라이딩 윈도우 방식으로 특징 개수 제한
            if len(self.local_tracks[best_match_id]['features']) > self.max_features:
                # 가장 오래된 특징부터 제거 (FIFO 방식)
                self.local_tracks[best_match_id]['features'] = self.local_tracks[best_match_id]['features'][-self.max_features:]
            
            print(f"Local ReID: Matched to existing track {best_match_id} (similarity: {best_similarity:.3f}, features: {len(self.local_tracks[best_match_id]['features'])})")
            return best_match_id
        else:
            # 새로운 로컬 ID 생성 (ByteTrack의 next_id() 사용)
            from ByteTrack.yolox.tracker.basetrack import BaseTrack
            local_id = BaseTrack.next_id()
            
            self.local_tracks[local_id] = {
                'features': [features],
                'last_seen': frame_id,
                'camera_id': camera_id,
                'last_bbox': bbox  # 위치 정보 저장
            }
            
            print(f"Local ReID: Created new track {local_id} (features: 1)")
            return local_id


def main(args):
    """메인 함수: Re-ID 모델 초기화, 워커 스레드 시작 및 GUI 처리"""
    reid_extractor = FeatureExtractor(
        model_name='osnet_ibn_x1_0',
        model_path= None,
        # device='cuda' 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    frame_queue = queue.Queue()  ##run_tracking 함수에서 넣어줌 
    stop_event = threading.Event()
    
    threads = []
    for video_path in args.videos: # 영상 개수만큼 스레드 생성
        # 스레드에 frame_queue와 stop_event 전달 / run_tracking 함수를 수행하는 스레드
        thread = threading.Thread(target=run_tracking, args=(video_path, args.yolo_model, reid_extractor, frame_queue, stop_event, args.videos.index(video_path), LocalReIDManager()), daemon=True)
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
    parser.add_argument('--videos', nargs='+', type=str, default=["test_video/KSEB02.mp4","test_video/KSEB03.mp4"], help='List of video file paths.')
    parser.add_argument('--yolo_model', type=str, default="models/weights/bestcctv.pt", help='Path to the YOLOv11 model file.')
    parser.add_argument('--reid_model', type=str, default=None, help='Path to the Re-ID model weights (.pth file). Leave empty to download pretrained.')
    
    cli_args = parser.parse_args()
    main(cli_args)