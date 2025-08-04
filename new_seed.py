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

# 경로 설정
try:
    sys.path.append('ByteTrack')
    sys.path.append('deep-person-reid-master')
    from yolox.tracker.byte_tracker import BYTETracker, STrack, TrackState
    from yolox.tracker.matching import iou_distance, linear_assignment
    from torchreid.utils.feature_extractor import FeatureExtractor
    from redis_global_reid_v2 import RedisGlobalReIDManagerV2
except ImportError as e:
    print(f"필수 라이브러리 로드 실패: {e}")
    sys.exit(1)

# np.float 호환성 문제 해결
if not hasattr(np, 'float'):
    np.float = float

class BYTETrackerWithReID(BYTETracker):
    def __init__(self, args, frame_rate=30, camera_id=0):
        super().__init__(args, frame_rate)
        self.camera_id = camera_id
        self.reid_thresh = 0.6
        self.global_id_mapping = {}  # {local_track_id: global_id}

    def update(self, dets, img_info, img_size, reid_extractor, global_reid_manager, frame_id):
        """글로벌 ReID를 사용한 업데이트"""
        online_targets = super().update(dets, img_info, img_size)
        return online_targets

def run_tracking(video_path, yolo_model_path, reid_extractor, frame_queue, stop_event, camera_id=0, global_reid_manager=None):
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
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_id += 1
        
        # 글로벌 ReID 매니저 프레임 업데이트
        if global_reid_manager is not None:
            global_reid_manager.update_frame(frame_id)

        # 프레임 리사이즈
        original_height, original_width = frame.shape[:2]
        target_width = 640
        scale = target_width / original_width
        target_height = int(original_height * scale)
        frame = cv2.resize(frame, (target_width, target_height))
        
        # 프레임 크기 저장 (사라지는 객체 감지용)
        frame_shape = frame.shape[:2]  # (height, width)
        
        # YOLO 탐지
        detection_results = model(frame, verbose=False, half=torch.cuda.is_available())[0]
        dets = []
        for box in detection_results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if classNames[cls_id].lower() in ["person", "saram"]:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append([x1, y1, x2, y2, conf])

        if len(dets) > 0:
            dets_array = np.array(dets, dtype=np.float32)
        else:
            dets_array = np.empty((0, 5), dtype=np.float32)

        # ByteTrack 추적
        online_targets = tracker.update(torch.tensor(dets_array), frame.shape[:2], frame.shape[:2], reid_extractor, global_reid_manager, frame_id)
        
        # 글로벌 ReID 처리
        if len(online_targets) > 0 and reid_extractor is not None and global_reid_manager is not None:
            crops = []
            bboxes = []
            for track in online_targets:
                x1, y1, x2, y2 = map(int, track.tlbr)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    crop = np.zeros((128, 64, 3), dtype=np.uint8)
                crops.append(crop)
                bboxes.append([x1, y1, x2, y2])
            
            if len(crops) > 0:
                features = reid_extractor(crops).cpu().numpy()
                matched_tracks = set()
                
                for i, track in enumerate(online_targets):
                    if i < len(features):
                        global_id = global_reid_manager.match_or_create(
                            features[i], bboxes[i], camera_id, frame_id, frame_shape, matched_tracks
                        )
                        if global_id is not None:
                            tracker.global_id_mapping[track.track_id] = global_id
                            print(f"Camera {camera_id}: Local ID {track.track_id} → Global ID {global_id}")
        
        # 결과 처리 및 그리기
        frame_detections_json = []
        for t in online_targets:
            if t.track_id not in tracked_ids:
                tracked_ids.add(t.track_id)
                person_count += 1
            
            xmin, ymin, xmax, ymax = map(int, t.tlbr)
            point_x = (xmin+xmax)/2
            point_y = ymin
            
            # 글로벌 ID 가져오기
            global_id = tracker.global_id_mapping.get(t.track_id, t.track_id)
            
            detection_data = {
                "camera_id": int(camera_id),
                "track_id": int(global_id),
                "bbox_xyxy": [point_x, point_y],
                "has_reid_feature": t.smooth_feat is not None if hasattr(t, 'smooth_feat') else False
            }
            frame_detections_json.append(detection_data)

            # 바운딩 박스 그리기
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f'ID:{global_id}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if frame_detections_json:
            print(f"--- {video_path} ---")
            print(json.dumps(frame_detections_json, indent=2))

        # 처리된 프레임을 큐에 삽입
        frame_queue.put((video_path, frame))

    cap.release()
    frame_queue.put((video_path, None))

def main(args):
    """메인 함수: Redis Global ReID V2 초기화 및 워커 스레드 시작"""
    # ReID 모델 초기화
    reid_extractor = FeatureExtractor(
        model_name='osnet_ibn_x1_0',
        model_path=None,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Redis Global ReID 매니저 V2 초기화 (모든 스레드가 공유)
    global_reid_manager = RedisGlobalReIDManagerV2(
        similarity_threshold=0.7,
        feature_ttl=300,
        max_features_per_camera=10,
        redis_host='localhost',
        redis_port=6379
    )
    
    frame_queue = queue.Queue()
    stop_event = threading.Event()
    
    # 워커 스레드 생성 (모든 스레드가 같은 global_reid_manager 공유)
    threads = []
    for video_path in args.videos:
        thread = threading.Thread(
            target=run_tracking, 
            args=(video_path, args.yolo_model, reid_extractor, frame_queue, stop_event, args.videos.index(video_path), global_reid_manager), 
            daemon=True
        )
        threads.append(thread)
        thread.start()

    # GUI 처리
    latest_frames = {}
    active_videos = set(args.videos)

    while active_videos:
        try:
            video_path, frame = frame_queue.get(timeout=0.1)

            if frame is None:
                active_videos.discard(video_path)
                if video_path in latest_frames:
                    del latest_frames[video_path]
                cv2.destroyWindow(f"Tracking - {video_path}")
                continue
            
            latest_frames[video_path] = frame

        except queue.Empty:
            pass

        for path, f in latest_frames.items():
            cv2.imshow(f"Tracking - {path}", f)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
    
    for thread in threads:
        thread.join()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv8 with ByteTrack and Redis Global Re-ID V2 for Multi-Video Tracking")
    parser.add_argument('--videos', nargs='+', type=str, default=["test_video/globaltest03.mp4","test_video/globaltest04.mp4"], help='List of video file paths.')
    parser.add_argument('--yolo_model', type=str, default="models/weights/bestcctv.pt", help='Path to the YOLOv11 model file.')
    parser.add_argument('--redis_host', type=str, default="localhost", help='Redis server host.')
    parser.add_argument('--redis_port', type=int, default=6379, help='Redis server port.')
    
    cli_args = parser.parse_args()
    main(cli_args) 