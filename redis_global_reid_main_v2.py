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
from point_transformer import transform_point

# 경로 설정
try:
    sys.path.append('ByteTrack')
    sys.path.append('deep-person-reid-master')
    sys.path.append('models/mapping')  # point_transformer 경로 추가
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

    tracker_args = argparse.Namespace(track_thresh=0.5, match_thresh=0.8, track_buffer=300, mot20=False)
    tracker = BYTETrackerWithReID(tracker_args, frame_rate=30, camera_id=camera_id)

    cap = cv2.VideoCapture(video_path)
    person_count = 0
    tracked_ids = set()
    frame_id = 0

    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print(f"[{video_path}] Video ended")
            break  # 영상이 끝나면 종료

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
        detection_results = model(frame, verbose=False, half=torch.cuda.is_available())[0] #verbose=False 출력메시지 최소화 /하나의 이미지만 처리중이지만 리스트로 반환하기때문에 [0]필요
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
                processed_crops = []
                for crop in crops:

                    normalized_crop = crop.astype(np.float32) / 255.0

                    processed_crops.append(normalized_crop)

                features = reid_extractor(processed_crops).cpu().numpy() #방금크롭한 특징 벡터들
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
            
            # 실제 좌표로 변환
            real_x, real_y = transform_point(point_x, point_y,
            [[0.000030, -0.000119, 0.043679],
            [-0.000115, -0.000221, 0.290054],
            [-0.000199, -0.000943, 1.000000]])
            
            # 글로벌 ID 가져오기
            global_id = tracker.global_id_mapping.get(t.track_id, t.track_id)
            
            # 디버깅: 매핑된 좌표 출력 (소수점 4자리)
            print(f"[DEBUG] Camera {camera_id}, Worker {global_id}: Image({point_x:.1f}, {point_y:.1f}) -> Real({real_x:.4f}, {real_y:.4f})")
            
            detection_data = {
                "cameraID": int(camera_id),
                "workerID": int(global_id),
                "position_X": real_x,
                "position_Y": real_y
            }
            frame_detections_json.append(detection_data)

            # 바운딩 박스 그리기
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f'ID:{global_id}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 매 프레임마다 JSON 데이터를 큐에 전송
        frame_queue.put((video_path, frame, frame_detections_json))

    cap.release()
    frame_queue.put((video_path, None, None))

#def run_tracking_realtime(video_path, yolo_model_path, reid_extractor, camera_id=0, global_reid_manager=None):
    # """실시간으로 매 프레임마다 결과를 yield하는 함수"""
    # model = YOLO(yolo_model_path, task="detect")
    # classNames = model.names

    # tracker_args = argparse.Namespace(track_thresh=0.5, match_thresh=0.8, track_buffer=300, mot20=False)
    # tracker = BYTETrackerWithReID(tracker_args, frame_rate=30, camera_id=camera_id)

    # cap = cv2.VideoCapture(video_path)
    # person_count = 0
    # tracked_ids = set()
    # frame_id = 0

    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         print(f"[{video_path}] Video ended")
    #         break

    #     frame_id += 1
        
    #     # 글로벌 ReID 매니저 프레임 업데이트
    #     if global_reid_manager is not None:
    #         global_reid_manager.update_frame(frame_id)

    #     # 프레임 리사이즈
    #     original_height, original_width = frame.shape[:2]
    #     target_width = 640
    #     scale = target_width / original_width
    #     target_height = int(original_height * scale)
    #     frame = cv2.resize(frame, (target_width, target_height))
        
    #     # 프레임 크기 저장 (사라지는 객체 감지용)
    #     frame_shape = frame.shape[:2]  # (height, width)
        
    #     # YOLO 탐지
    #     detection_results = model(frame, verbose=False, half=torch.cuda.is_available())[0]
    #     dets = []
    #     for box in detection_results.boxes:
    #         cls_id = int(box.cls[0])
    #         conf = float(box.conf[0])
    #         if classNames[cls_id].lower() in ["person", "saram"]:
    #             x1, y1, x2, y2 = box.xyxy[0].tolist()
    #             dets.append([x1, y1, x2, y2, conf])

    #     if len(dets) > 0:
    #         dets_array = np.array(dets, dtype=np.float32)
    #     else:
    #         dets_array = np.empty((0, 5), dtype=np.float32)

    #     # ByteTrack 추적
    #     online_targets = tracker.update(torch.tensor(dets_array), frame.shape[:2], frame.shape[:2], reid_extractor, global_reid_manager, frame_id)
        
    #     # 글로벌 ReID 처리
    #     if len(online_targets) > 0 and reid_extractor is not None and global_reid_manager is not None:
    #         crops = []
    #         bboxes = []
    #         for track in online_targets:
    #             x1, y1, x2, y2 = map(int, track.tlbr)
    #             crop = frame[y1:y2, x1:x2]
    #             if crop.size == 0:
    #                 crop = np.zeros((128, 64, 3), dtype=np.uint8)
    #             crops.append(crop)
    #             bboxes.append([x1, y1, x2, y2])
            
    #         if len(crops) > 0:
    #             processed_crops = []
    #             for crop in crops:
    #                 normalized_crop = crop.astype(np.float32) / 255.0
    #                 processed_crops.append(normalized_crop)

    #             features = reid_extractor(processed_crops).cpu().numpy()
    #             matched_tracks = set()
                
    #             for i, track in enumerate(online_targets):
    #                 if i < len(features):
    #                     global_id = global_reid_manager.match_or_create(
    #                         features[i], bboxes[i], camera_id, frame_id, frame_shape, matched_tracks
    #                     )
    #                     if global_id is not None:
    #                         tracker.global_id_mapping[track.track_id] = global_id
    #                         print(f"Camera {camera_id}: Local ID {track.track_id} → Global ID {global_id}")
        
    #     # 결과 처리 및 yield
    #     frame_detections_json = []
    #     for t in online_targets:
    #         if t.track_id not in tracked_ids:
    #             tracked_ids.add(t.track_id)
    #             person_count += 1
            
    #         xmin, ymin, xmax, ymax = map(int, t.tlbr)
    #         point_x = np.float64((xmin+xmax)/2)
    #         point_y = np.float64(ymin)
            
    #         # 실제 좌표로 변환
    #         real_x, real_y = transform_point(point_x, point_y,
    #         [[0.000030, -0.000119, 0.043679],
    #         [-0.000115, -0.000221, 0.290054],
    #         [-0.000199, -0.000943, 1.000000]])
            
    #         # 글로벌 ID 가져오기
    #         global_id = tracker.global_id_mapping.get(t.track_id, t.track_id)
            
    #         # 디버깅: 매핑된 좌표 출력 (소수점 4자리)
    #         print(f"[DEBUG] Camera {camera_id}, Worker {global_id}: Image({point_x:.1f}, {point_y:.1f}) -> Real({real_x:.4f}, {real_y:.4f})")
            
    #         detection_data = {
    #             "cameraID": int(camera_id),
    #             "workerID": int(global_id),
    #             "position_X": real_x,
    #             "position_Y": real_y,
    #             "frame_id": frame_id
    #         }
    #         frame_detections_json.append(detection_data)

    #     # 매 프레임마다 결과 yield
    #     yield frame_detections_json

    # cap.release()

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
        similarity_threshold=0.5,
        feature_ttl=3000,  # 100초
        max_features_per_camera=10,
        redis_host='localhost',
        redis_port=6379,
        frame_rate=30  # 30fps
    )
    
    frame_queue = queue.Queue()
    stop_event = threading.Event()
    
    # 워커 스레드 생성 (모든 스레드가 같은 global_reid_manager 공유)
    threads = []
    for video_path in args.videos:
        thread = threading.Thread(
            target=run_tracking, 
            args=(video_path, args.yolo_model, reid_extractor, frame_queue, stop_event, args.videos.index(video_path), global_reid_manager, None),  # 모든 프레임 처리
            daemon=True
        )
        threads.append(thread)
        thread.start()

    # GUI 처리
    latest_frames = {}
    active_videos = set(args.videos)

    # JSON 데이터 수집용 딕셔너리
    latest_detections = {}
    
    # 매 프레임마다 실시간 처리
    frame_count = 0
    max_frames = 300  # 최대 300프레임 (약 10초) 처리
    
    while active_videos and frame_count < max_frames:
        try:
            video_path, frame, detections = frame_queue.get(timeout=0.1)

            if frame is None:
                active_videos.discard(video_path)
                if video_path in latest_detections:
                    del latest_detections[video_path]
                continue
            
            frame_count += 1
            if detections:
                latest_detections[video_path] = detections

        except queue.Empty:
            continue
    
    # 모든 카메라의 감지 결과를 합쳐서 반환
    all_detections = []
    for detections in latest_detections.values():
        all_detections.extend(detections)
    
    # 스레드 정리
    stop_event.set()
    for thread in threads:
        thread.join(timeout=1.0)  # 1초 타임아웃으로 정리
    
    return all_detections

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv8 with ByteTrack and Redis Global Re-ID V2 for Multi-Video Tracking")
    parser.add_argument('--videos', nargs='+', type=str, default=["test_video/KSEB02.mp4","test_video/KSEB03.mp4"], help='List of video file paths.')
    parser.add_argument('--yolo_model', type=str, default="models/weights/bestcctv.pt", help='Path to the YOLOv11 model file.')
    parser.add_argument('--redis_host', type=str, default="localhost", help='Redis server host.')
    parser.add_argument('--redis_port', type=int, default=6379, help='Redis server port.')
    
    cli_args = parser.parse_args()
    main(cli_args) 