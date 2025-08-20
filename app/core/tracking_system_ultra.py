import cv2
import numpy as np
import threading
import queue
import time
import json
from pathlib import Path
from app.image_processor import ImageProcessor


# 내부 모듈 import (새 구조)
from app.detector.ultralytics_tracker import UltralyticsTrackerManager
from app.reid.reid_manager import GlobalReIDManager
from app.reid.redis_handler import FeatureStoreRedisHandler
from app.reid.similarity import FeatureSimilarityCalculator
from app.config import settings
from app.image_processor import ImageProcessor
from app.result.performance_logger import PerformanceLogger
from app.ppe.ppe_detector import PPEDetector
from app.io.backend_client import BackendClient
from app.models.mapping.point_transformer import transform_point
from app.core.matching_cache_manager import MatchingCacheManager


class IntegratedTrackingSystemUltra:
    """
    Ultralytics Tracking을 사용하는 ReID 기능과 호모그래피 기능,
    PPE 탐지 기능을 통합한 객체 추적 시스템
    """
    
    def __init__(self, video_paths=None, model_path=None, tracker_config=None, 
                 redis_conf=None, reid_conf=None, calibration_files=None,
                 backend_url=None, ppe_model_path=None):
        
        # 설정 로드
        self.model_path = model_path or settings.YOLO_MODEL_PATH
        self.tracker_config = tracker_config or settings.TRACKER_CONFIG
        redis_conf = redis_conf or settings.REDIS_CONFIG
        reid_conf = reid_conf or settings.REID_CONFIG
        
        # 비디오 경로
        self.video_paths = video_paths or settings.VIDEO_INPUT_PATHS
        
        # PPE 탐지기 초기화
        self.ppe_detector = PPEDetector(ppe_model_path or "models/weights/best_yolo11n.pt")
        
        # 백엔드 클라이언트 초기화
        self.backend_client = BackendClient(backend_url or "http://localhost:5000")
        
        # Redis 핸들러 초기화
        redis_handler = FeatureStoreRedisHandler(
            redis_host=redis_conf.get("host", "localhost"),
            redis_port=redis_conf.get("port", 6379)
        )
        similarity = FeatureSimilarityCalculator()
        self.reid = GlobalReIDManager(
            redis_handler,
            similarity,
            similarity_threshold=reid_conf.get("threshold", 0.7)
        )
        
        # 매핑/캐시/결과 저장소
        self.local_to_global_mapping = {}
        self.global_to_cameras = {}
        self.image_processor = ImageProcessor()
        self.thread_performance_loggers = {}
        self.tracking_results = {}
        self.ppe_violation_history = {}
        self.matching_cache_manager = MatchingCacheManager()
        
        print("Integrated Tracking System (Ultralytics) initialized")
        print(f"Videos: {self.video_paths}")
        print(f"Model: {self.model_path}")
        print(f"PPE Model: {ppe_model_path or 'models/weights/best_yolo11n.pt'}")
        print(f"Backend URL: {backend_url or 'http://localhost:5000'}")
        
        # 호모그래피 매트릭스 출력
        print(f"Homography matrices available for cameras: {list(settings.HOMOGRAPHY_MATRICES.keys())}")
        for camera_id in settings.HOMOGRAPHY_MATRICES.keys():
            matrix = np.array(settings.HOMOGRAPHY_MATRICES[camera_id])
            print(f"  Camera {camera_id}: Matrix shape {matrix.shape}")
    
    def add_mapping(self, camera_id: str, local_id: int, global_id: int):
        """매핑 추가 시 자동으로 중복 제거 - O(1) 최적화"""
        camera_id = str(camera_id)
        local_id = int(local_id)
        global_id = int(global_id)
        
        if (global_id in self.global_to_cameras and 
            camera_id in self.global_to_cameras[global_id] and
            local_id <= self.global_to_cameras[global_id][camera_id]):
            return
        
        if (global_id in self.global_to_cameras and 
            camera_id in self.global_to_cameras[global_id]):
            old_key = f"{camera_id}_{self.global_to_cameras[global_id][camera_id]}"
            self.local_to_global_mapping.pop(old_key, None)
        
        camera_key = f"{camera_id}_{local_id}"
        self.local_to_global_mapping[camera_key] = global_id
        
        if global_id not in self.global_to_cameras:
            self.global_to_cameras[global_id] = {}
        self.global_to_cameras[global_id][camera_id] = local_id

    def _calculate_best_similarity(self, features: np.ndarray, camera_id: str) -> float:
        try:
            candidates = self.reid.redis.get_candidate_features_by_camera(camera_id)
            if not candidates:
                return 0.0
            best_similarity = 0.0
            for global_id, candidate_data in candidates.items():
                candidate_features = candidate_data['features']
                if len(candidate_features) > 0:
                    features_array = np.array(candidate_features)
                    if len(features_array) == 1:
                        weighted_average = features_array[0]
                    else:
                        weight_start = settings.REID_CONFIG["same_camera"]["weight_start"]
                        weight_end = settings.REID_CONFIG["same_camera"]["weight_end"]
                        weights = np.linspace(weight_start, weight_end, len(features_array))
                        weights = weights / np.sum(weights)
                        weighted_average = np.average(features_array, axis=0, weights=weights)
                    similarity = self.reid.similarity.calculate_similarity(features, weighted_average, f"sim_{camera_id}_{global_id}")
                    best_similarity = max(best_similarity, similarity)
            return best_similarity
        except Exception as e:
            print(f"Error calculating best similarity: {e}")
            return 0.0

    def create_detector_for_thread(self):
        return UltralyticsTrackerManager(self.model_path, self.tracker_config)
    
    def process_frame_with_reid_and_homography(self, frame, frame_id, camera_id, detector):
        if camera_id not in self.thread_performance_loggers:
            self.thread_performance_loggers[camera_id] = PerformanceLogger(output_dir=f"result/camera_{camera_id}")
        logger = self.thread_performance_loggers[camera_id]
        logger.start_frame_timing(frame_id, camera_id)
        self.reid.update_frame(frame_id)
        logger.start_detection_timing()
        logger.start_tracking_timing()
        track_list = detector.detect_and_track(frame, frame_id)
        logger.end_detection_timing()
        logger.end_tracking_timing()
        
        frame_matched_tracks = set()
        frame_detections, frame_workers = [], []
        logger.set_object_count(len(track_list))
        
        track_similarities = []
        for track in track_list:
            local_id = track["track_id"]
            bbox = track["bbox"]
            crop, feature = self.image_processor.process_track_for_reid(frame, track)
            camera_key = f"{camera_id}_{local_id}"
            if camera_key in self.local_to_global_mapping:
                track_similarities.append((track, feature, 1.0, True))
            else:
                best_similarity = self._calculate_best_similarity(feature, camera_id)
                track_similarities.append((track, feature, best_similarity, False))
        
        track_similarities.sort(key=lambda x: x[2], reverse=True)
        
        for track, feature, similarity, is_existing in track_similarities:
            local_id = track["track_id"]
            bbox = track["bbox"]
            camera_key = f"{camera_id}_{local_id}"
            if is_existing:
                global_id = self.local_to_global_mapping[camera_key]
                logger.start_same_camera_reid_timing()
                self.reid._update_track_camera(global_id, feature, bbox, str(camera_id), frame_id, local_id)
                logger.end_same_camera_reid_timing()
            else:
                logger.start_cross_camera_reid_timing()
                global_id = self.reid.match_or_create(
                    features=feature, bbox=bbox, camera_id=str(camera_id),
                    frame_id=frame_id, frame_shape=frame.shape[:2],
                    matched_tracks=frame_matched_tracks, local_track_id=local_id
                )
                logger.end_cross_camera_reid_timing()
                if global_id is None:
                    global_id = local_id
                else:
                    self.add_mapping(str(camera_id), local_id, global_id)
            frame_matched_tracks.add(global_id)
            
            x1, y1, x2, y2, point_x, point_y = self.image_processor.get_bbox_coordinates(bbox)
            try:
                homography_matrix = np.array(settings.HOMOGRAPHY_MATRICES[camera_id])
                real_x, real_y = transform_point(point_x, point_y, homography_matrix)
            except Exception as e:
                print(f"Error in coordinate transformation for camera {camera_id}: {e}")
                real_x, real_y = 0, 0
            
            ppe_violations = self.ppe_detector.detect_ppe_violations(frame, bbox)
            raw_types = [v['type'] for v in ppe_violations] if ppe_violations else []
            mapped = []
            for t in raw_types:
                tl = str(t).lower()
                if "helmet" in tl:
                    mapped.append("helmet_missing")
                elif "vest" in tl:
                    mapped.append("vest_missing")
            ppe_list = sorted(list(set(mapped)))
            
            if ppe_list:
                violation_key = f"{camera_id}_{global_id}_{frame_id}"
                if violation_key not in self.ppe_violation_history:
                    self.ppe_violation_history[violation_key] = True
                    worker_entry = {
                        "worker_id": f"W{global_id:03d}",
                        "zone_id": f"Z{camera_id:02d}",
                        "violations": { "ppe": ppe_list, "roi": [] }
                    }
                    frame_workers.append(worker_entry)
                    print(f"[PPE VIOLATION] Worker {global_id} in Camera {camera_id}: {ppe_list}")
            
            detection_data = {
                "cameraID": int(camera_id),
                "workerID": int(global_id),
                "position_X": real_x,
                "position_Y": real_y,
                "frame_id": frame_id,
                "local_id": local_id,
                "bbox": [x1, y1, x2, y2],
                "image_coords": [point_x, point_y],
                "ppe_violations": [v['type'] for v in ppe_violations] if ppe_violations else []
            }
            frame_detections.append(detection_data)
            
            color = (0, 0, 255) if ppe_violations else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'ID:{global_id}', (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"({real_x:.1f}, {real_y:.1f})", (x1, y2 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            if ppe_violations:
                cv2.putText(frame, f"PPE: {', '.join([v['type'] for v in ppe_violations])}",
                           (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        logger.log_frame_performance()
        return frame, frame_detections, frame_workers
    
    def send_detections_to_backend(self, detections, workers_items):
        if detections:
            worker_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "workers": []
            }
            for det in detections:
                worker_info = {
                    "worker_id": str(det["workerID"]),
                    "zone_id": f"Z{det['cameraID']:02d}",
                    "x": float(det["position_X"]),
                    "y": float(det["position_Y"]),
                    "product_count": 0,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                worker_data["workers"].append(worker_info)
            self.backend_client.send_worker_data(worker_data)
        
        if workers_items:
            iso_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            payload = { "timestamp": iso_ts, "workers": workers_items }
            self.backend_client.send_violation_data(payload)
    
    def run_video_thread(self, video_path, camera_id, frame_buffer, sync_event, stop_event):
        detector = self.create_detector_for_thread()
        cap = cv2.VideoCapture(video_path)
        frame_id = 0
        processed_frame_id = 0
        target_fps = 30
        frame_skip = 30/target_fps
        while cap.isOpened() and not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print(f"[{video_path}] Video ended")
                break
            frame_id += 1
            if frame_id % frame_skip != 0:
                continue
            processed_frame_id += 1
            processed_frame, detections, workers_items = self.process_frame_with_reid_and_homography(
                frame, processed_frame_id, camera_id, detector
            )
            self.send_detections_to_backend(detections, workers_items)
            frame_data = {
                'video_path': video_path,
                'frame': processed_frame,
                'detections': detections,
                'violations': workers_items,
                'frame_id': processed_frame_id
            }
            frame_buffer.put(frame_data)
            sync_event.wait(0.1)
            sync_event.clear()
        cap.release()
        frame_buffer.put({'video_path': video_path, 'frame': None, 'detections': None, 'violations': None, 'frame_id': -1})
    
    def run_multi_video_tracking(self):
        stop_event = threading.Event()
        sync_event = threading.Event()
        frame_buffers = {video_path: queue.Queue(maxsize=1) for video_path in self.video_paths}
        collector_threads = []
        for i, video_path in enumerate(self.video_paths):
            thread = threading.Thread(
                target=self.run_video_thread,
                args=(video_path, i, frame_buffers[video_path], sync_event, stop_event),
                daemon=True
            )
            collector_threads.append(thread)
            thread.start()
        active_videos = set(self.video_paths)
        latest_detections, latest_violations = {}, {}
        frame_count, max_frames = 0, 3000
        while active_videos and frame_count < max_frames:
            all_frames_ready, current_frames = True, {}
            for video_path in active_videos.copy():
                try:
                    frame_data = frame_buffers[video_path].get(timeout=0.01)
                    if frame_data['frame'] is None:
                        active_videos.discard(video_path)
                        latest_detections.pop(video_path, None)
                        latest_violations.pop(video_path, None)
                        continue
                    current_frames[video_path] = frame_data
                except queue.Empty:
                    all_frames_ready = False
                    break
            if not all_frames_ready:
                time.sleep(0.01)
                continue
            frame_count += 1
            print(f"[SYNC] Processing frame {frame_count} for all videos")
            for video_path, frame_data in current_frames.items():
                if frame_data['detections']:
                    latest_detections[video_path] = frame_data['detections']
                if frame_data['violations']:
                    latest_violations[video_path] = frame_data['violations']
            sync_event.set()
        stop_event.set()
        for thread in collector_threads:
            thread.join(timeout=1.0)
        return latest_detections, latest_violations
    
    def save_tracking_results(self, output_file="tracking_results_ultra.json"):
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "videos": self.video_paths,
            "model": self.model_path,
            "ppe_model": self.ppe_detector.model_path,
            "tracker": "Ultralytics",
            "total_detections": len(self.tracking_results),
            "results": self.tracking_results
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Tracking results saved to: {output_file}")
    
    def print_performance_summary(self):
        print("\n" + "="*60)
        print("📊 INTEGRATED TRACKING PERFORMANCE SUMMARY (Ultralytics)")
        print("="*60)
        for camera_id, thread_logger in self.thread_performance_loggers.items():
            print(f"\n📊 Camera {camera_id} Performance Summary:")
            thread_logger.print_summary()
        print("="*60)
