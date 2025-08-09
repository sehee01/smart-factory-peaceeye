# detector_manager.py
from ultralytics import YOLO
from detector.bytetrack_processor import ByteTrackProcessor
import cv2
import numpy as np
import torch
from typing import List, Dict, Any


class ByteTrackDetectorManager:
    """
    YOLO 탐지와 ByteTrack 추적을 통합 관리하는 매니저
    원본의 복잡한 기능을 포함: 프레임 리사이즈, 클래스 필터링, 바운딩 박스 변환
    """
    
    def __init__(self, model_path: str, tracker_config: dict):
        # ① YOLO 모델 로드 → self.model 생성
        self.model = YOLO(model_path, task="detect")
        self.class_names = self.model.names
        
        # ② BYTETracker 래퍼
        self.bt = ByteTrackProcessor(tracker_config)
        self.frame_rate = tracker_config.get("frame_rate", 30)
        
        # ③ 설정
        self.target_width = tracker_config.get("target_width", 640)
        self.person_classes = ["person", "saram"]  # 탐지할 클래스들
        
        # 원본 프레임 크기 저장용
        self.original_frame_shape = None

    def detect_and_track(self, frame, frame_id: int):
        """
        프레임에서 객체 탐지 및 추적 수행
        
        Args:
            frame: 입력 프레임
            frame_id: 프레임 ID
            
        Returns:
            track_list: 추적 결과 리스트
        """
        # 원본 프레임 크기 저장
        self.original_frame_shape = frame.shape[:2]  # (height, width)
        
        # 1. 프레임 리사이즈 (원본 기능 복원)
        resized_frame = self._resize_frame(frame)
        
        # 2. YOLO 탐지
        detection_results = self._run_detection(resized_frame)
        
        # 3. 탐지 결과 변환
        dets = self._convert_detections(detection_results)
        
        # 4. ByteTrack 추적
        track_list = self.bt.update_tracks(dets, resized_frame, frame_id, fps=self.frame_rate)
        
        # 5. 좌표를 원본 프레임 크기로 변환
        track_list = self._convert_coordinates_to_original(track_list)
        
        return track_list

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        프레임을 목표 크기로 리사이즈 (원본 기능 복원)
        
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

    def _run_detection(self, frame: np.ndarray) -> Any:
        """
        YOLO 모델로 객체 탐지 수행
        
        Args:
            frame: 입력 프레임
            
        Returns:
            detection_results: 탐지 결과
        """
        # YOLO 추론 (GPU 사용 가능시 half precision 사용)
        results = self.model(frame, verbose=False, half=torch.cuda.is_available())[0]
        return results

    def _convert_detections(self, detection_results: Any) -> List[List[float]]:
        """
        YOLO 탐지 결과를 ByteTrack 형식으로 변환
        
        Args:
            detection_results: YOLO 탐지 결과
            
        Returns:
            dets: ByteTrack 형식의 탐지 결과 [[x1, y1, x2, y2, conf], ...]
        """
        dets = []
        boxes = detection_results.boxes
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # 사람 클래스만 필터링 (원본 기능 복원)
                class_name = self.class_names[cls_id].lower()
                if class_name in self.person_classes:
                    # xyxy 형식으로 변환 (원본과 동일)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    dets.append([x1, y1, x2, y2, conf])

        return dets

    def _convert_coordinates_to_original(self, track_list: List[Dict]) -> List[Dict]:
        """
        리사이즈된 프레임의 좌표를 원본 프레임 크기로 변환
        
        Args:
            track_list: 리사이즈된 프레임에서의 추적 결과
            
        Returns:
            converted_track_list: 원본 프레임 크기로 변환된 추적 결과
        """
        if self.original_frame_shape is None:
            return track_list
            
        original_height, original_width = self.original_frame_shape
        target_width = self.target_width
        scale = target_width / original_width
        target_height = int(original_height * scale)
        
        # 스케일 팩터 계산 (원본 크기 / 리사이즈된 크기)
        scale_x = original_width / target_width
        scale_y = original_height / target_height
        
        # 디버그 모드일 때만 로그 출력 (기본적으로는 출력하지 않음)
        debug_mode = False  # 필요시 True로 변경
        if debug_mode:
            print(f"[DEBUG] 좌표 변환 정보:")
            print(f"  원본 크기: {original_width}x{original_height}")
            print(f"  리사이즈 크기: {target_width}x{target_height}")
            print(f"  스케일 팩터: X={scale_x:.3f}, Y={scale_y:.3f}")
        
        converted_track_list = []
        for i, track in enumerate(track_list):
            bbox = track["bbox"]
            x1, y1, x2, y2 = bbox
            
            # 좌표를 원본 크기로 변환
            original_x1 = x1 * scale_x
            original_y1 = y1 * scale_y
            original_x2 = x2 * scale_x
            original_y2 = y2 * scale_y
            
            if debug_mode:
                print(f"  Track {track['track_id']}: 리사이즈 좌표({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) -> 원본 좌표({original_x1:.1f},{original_y1:.1f},{original_x2:.1f},{original_y2:.1f})")
            
            converted_track = {
                "track_id": track["track_id"],
                "bbox": [original_x1, original_y1, original_x2, original_y2]
            }
            converted_track_list.append(converted_track)
            
        return converted_track_list

    def get_frame_shape(self, frame: np.ndarray) -> tuple:
        """
        프레임 크기 반환 (사라지는 객체 감지용)
        
        Args:
            frame: 프레임
            
        Returns:
            frame_shape: (height, width) 튜플
        """
        return frame.shape[:2]

    def filter_by_class(self, detections: List[List[float]], class_names: List[str] = None) -> List[List[float]]:
        """
        클래스별로 탐지 결과 필터링
        
        Args:
            detections: 탐지 결과
            class_names: 필터링할 클래스명 리스트
            
        Returns:
            filtered_detections: 필터링된 탐지 결과
        """
        if class_names is None:
            class_names = self.person_classes
            
        # 현재는 단순히 모든 탐지 결과를 반환
        # 실제로는 클래스 정보를 포함한 탐지 결과에서 필터링 필요
        return detections

    def convert_to_tlbr(self, detections: List[List[float]]) -> List[List[float]]:
        """
        탐지 결과를 tlbr (top-left, bottom-right) 형식으로 변환
        
        Args:
            detections: 탐지 결과
            
        Returns:
            tlbr_detections: tlbr 형식의 탐지 결과
        """
        return detections
