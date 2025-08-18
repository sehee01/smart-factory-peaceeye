# ultralytics_tracker.py
from ultralytics import YOLO
import cv2
import numpy as np
import torch
from typing import List, Dict, Any


class UltralyticsTrackerManager:
    """
    Ultralytics YOLO의 내장 tracking 기능을 사용하는 매니저
    ByteTrack 대신 Ultralytics의 내장 tracker 사용
    """
    
    def __init__(self, model_path: str, tracker_config: dict):
        # YOLO 모델 로드 (tracking 모드)
        self.model = YOLO(model_path, task="detect")
        self.class_names = self.model.names
        
        # Tracking 설정
        self.tracker_config = tracker_config
        self.target_width = tracker_config.get("target_width", 640)
        self.person_classes = ["person", "saram"]  # 탐지할 클래스들
        
        # 원본 프레임 크기 저장용
        self.original_frame_shape = None
        
        # Tracking 상태 초기화
        self.track_history = {}
        self.max_history_length = tracker_config.get("track_buffer", 30)

    def detect_and_track(self, frame, frame_id: int):
        """
        프레임에서 객체 탐지 및 추적 수행 (Ultralytics 내장 tracking 사용)
        
        Args:
            frame: 입력 프레임
            frame_id: 프레임 ID
            
        Returns:
            track_list: 추적 결과 리스트
        """
        # 원본 프레임 크기 저장
        self.original_frame_shape = frame.shape[:2]  # (height, width)
        
        # 1. 프레임 리사이즈
        resized_frame = self._resize_frame(frame)
        
        # 2. YOLO tracking 실행 (내장 tracker 사용)
        results = self._run_tracking(resized_frame)
        
        # 3. 결과를 표준 형식으로 변환
        track_list = self._convert_tracking_results(results, frame_id)
        
        # 4. 좌표를 원본 프레임 크기로 변환
        track_list = self._convert_coordinates_to_original(track_list)
        
        return track_list

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        프레임을 목표 크기로 리사이즈
        
        Args:
            frame: 원본 프레임
            
        Returns:
            resized_frame: 리사이즈된 프레임
        """
        original_height, original_width = frame.shape[:2]
        target_width = self.target_width
        scale = target_width / original_width
        target_height = int(original_height * scale)
        
        resized_frame = cv2.resize(frame, (target_width, target_height))
        return resized_frame

    def _run_tracking(self, frame: np.ndarray) -> Any:
        """
        YOLO 모델로 객체 탐지 및 추적 수행 (내장 tracker 사용)
        
        Args:
            frame: 입력 프레임
            
        Returns:
            results: tracking 결과
        """
        # YOLO tracking 추론 (GPU 사용 가능시 half precision 사용)
        results = self.model.track(
            frame, 
            verbose=False, 
            half=torch.cuda.is_available(),
            persist=True,  # 프레임 간 추적 상태 유지
            tracker="bytetrack.yaml"  # Ultralytics의 ByteTrack 구현 사용
        )[0]
        return results

    def _convert_tracking_results(self, results: Any, frame_id: int) -> List[Dict]:
        """
        Ultralytics tracking 결과를 표준 형식으로 변환
        
        Args:
            results: Ultralytics tracking 결과
            frame_id: 프레임 ID
            
        Returns:
            track_list: 표준 형식의 추적 결과
        """
        track_list = []
        
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2)
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for i, (box, track_id, conf, class_id) in enumerate(zip(boxes, track_ids, confidences, class_ids)):
                # person 클래스만 필터링
                class_name = self.class_names[class_id]
                if class_name.lower() in [cls.lower() for cls in self.person_classes]:
                    x1, y1, x2, y2 = box
                    
                    track_list.append({
                        "track_id": int(track_id),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(conf),
                        "class_id": int(class_id),
                        "class_name": class_name
                    })
                    
                    # Track history 업데이트
                    self._update_track_history(track_id, [x1, y1, x2, y2], frame_id)
        
        return track_list

    def _update_track_history(self, track_id: int, bbox: List[float], frame_id: int):
        """
        추적 히스토리 업데이트
        
        Args:
            track_id: 추적 ID
            bbox: 바운딩 박스 [x1, y1, x2, y2]
            frame_id: 프레임 ID
        """
        if track_id not in self.track_history:
            self.track_history[track_id] = []
        
        # 중심점 계산
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        self.track_history[track_id].append({
            'frame_id': frame_id,
            'center': [center_x, center_y],
            'bbox': bbox
        })
        
        # 히스토리 길이 제한
        if len(self.track_history[track_id]) > self.max_history_length:
            self.track_history[track_id] = self.track_history[track_id][-self.max_history_length:]

    def _convert_coordinates_to_original(self, track_list: List[Dict]) -> List[Dict]:
        """
        좌표를 원본 프레임 크기로 변환
        
        Args:
            track_list: 리사이즈된 프레임 기준 추적 결과
            
        Returns:
            track_list: 원본 프레임 크기로 변환된 추적 결과
        """
        if self.original_frame_shape is None:
            return track_list
        
        original_height, original_width = self.original_frame_shape
        target_width = self.target_width
        scale = original_width / target_width
        target_height = int(original_height / scale)
        
        for track in track_list:
            bbox = track["bbox"]
            x1, y1, x2, y2 = bbox
            
            # 좌표 변환
            x1_orig = x1 * scale
            y1_orig = y1 * scale
            x2_orig = x2 * scale
            y2_orig = y2 * scale
            
            track["bbox"] = [x1_orig, y1_orig, x2_orig, y2_orig]
        
        return track_list

    def get_track_history(self, track_id: int) -> List[Dict]:
        """
        특정 track_id의 히스토리 반환
        
        Args:
            track_id: 추적 ID
            
        Returns:
            history: 추적 히스토리
        """
        return self.track_history.get(track_id, [])

    def clear_track_history(self):
        """모든 추적 히스토리 초기화"""
        self.track_history.clear()
