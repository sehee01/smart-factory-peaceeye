import cv2
import numpy as np
import torch
import sys
from pathlib import Path

# torchreid 모듈 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TORCHREID_PATH = str(PROJECT_ROOT / "deep-person-reid-master")
if TORCHREID_PATH not in sys.path:
    sys.path.insert(0, TORCHREID_PATH)

from torchreid.utils.feature_extractor import FeatureExtractor
from app.config import settings


class ImageProcessor:
    """
    이미지 크롭 및 feature 추출을 담당하는 클래스
    단일 책임 원칙에 따라 이미지 처리 로직을 분리
    """
    
    def __init__(self):
        # Feature Extractor 초기화 (설정에서 가져온 값 사용)
        device = settings.FEATURE_EXTRACTOR_CONFIG["device"]
        if device == "auto":
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.feature_extractor = FeatureExtractor(
            model_name=settings.FEATURE_EXTRACTOR_CONFIG["model_name"],
            model_path=settings.FEATURE_EXTRACTOR_CONFIG["model_path"],
            device=device
        )
    
    def calculate_iou(self, bbox1: list, bbox2: list) -> float:
        """
        두 바운딩 박스 간의 IoU(Intersection over Union) 계산
        
        Args:
            bbox1: 첫 번째 바운딩 박스 [x1, y1, x2, y2]
            bbox2: 두 번째 바운딩 박스 [x1, y1, x2, y2]
            
        Returns:
            iou: IoU 값 (0~1)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 교집합 영역 계산
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 합집합 영역 계산
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def detect_overlap(self, current_bbox: list, all_tracks: list, current_track_id: int, overlap_threshold: float = 0.4) -> bool:
        """
        현재 트랙이 다른 트랙과 겹치는지 감지
        
        Args:
            current_bbox: 현재 트랙의 바운딩 박스
            all_tracks: 모든 트랙 리스트
            current_track_id: 현재 트랙의 ID
            overlap_threshold: 겹침 임계값 (기본값: 0.4)
            
        Returns:
            bool: 겹침 여부
        """
        for track in all_tracks:
            if track["track_id"] == current_track_id:
                continue
            
            other_bbox = track["bbox"]
            iou = self.calculate_iou(current_bbox, other_bbox)
            
            if iou > overlap_threshold:
                # print(f"[ImageProcessor] Overlap detected: Track {current_track_id} overlaps with Track {track['track_id']} (IoU: {iou:.3f})")
                return True
        
        return False
    
    def crop_bbox_from_frame(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """
        프레임에서 바운딩 박스 영역을 크롭
        
        Args:
            frame: 입력 프레임
            bbox: 바운딩 박스 [x1, y1, x2, y2]
            
        Returns:
            cropped_image: 크롭된 이미지
        """
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        return crop
    
    def extract_feature(self, crop_img: np.ndarray) -> np.ndarray:
        """
        크롭된 이미지에서 feature 추출
        전문적인 feature extractor를 우선 사용하고, 실패 시 단순한 방법 사용
        
        Args:
            crop_img: 크롭된 이미지
            
        Returns:
            feature: 추출된 feature 벡터
        """
        if crop_img.size == 0:
            return np.zeros(512)  # osnet_ibn_x1_0의 feature dimension
        
        if self.feature_extractor is not None:
            # 전문적인 feature extractor 사용
            return self._extract_feature_with_extractor(crop_img)
        else:
            # 단순한 RGB 평균 feature (fallback)
            return self._extract_feature_simple(crop_img)
    
    def _extract_feature_with_extractor(self, crop_img: np.ndarray) -> np.ndarray:
        """
        전문적인 feature extractor를 사용한 feature 추출 (패딩 없이 resize만 사용)
        
        Args:
            crop_img: 크롭된 이미지
            
        Returns:
            feature: 추출된 feature 벡터
        """
        if crop_img.size == 0:
            return np.zeros(512)  # osnet_ibn_x1_0의 feature dimension
        
        # 패딩 없이 단순 resize
        target_size = settings.FEATURE_EXTRACTOR_CONFIG["target_size"]
        resized_crop = cv2.resize(crop_img, target_size)
        normalized_crop = resized_crop.astype(np.float32) / 255.0
        
        # Feature 추출
        with torch.no_grad():
            feature = self.feature_extractor([normalized_crop]).cpu().numpy()
            return feature.flatten()
    
    def _extract_feature_simple(self, crop_img: np.ndarray) -> np.ndarray:
        """
        단순한 RGB 평균 feature 추출 (fallback)
        
        Args:
            crop_img: 크롭된 이미지
            
        Returns:
            feature: 추출된 feature 벡터
        """
        if crop_img.size == 0:
            return np.zeros(512)
        feature = crop_img.mean(axis=(0, 1))  # RGB 평균
        return feature / 255.0
    
    def process_track_for_reid(self, frame: np.ndarray, track: dict, all_tracks: list = None) -> tuple:
        """
        트랙 정보를 받아서 크롭 및 feature 추출을 수행
        겹침이 감지되면 crop을 하지 않고 None을 반환
        
        Args:
            frame: 입력 프레임
            track: 트랙 정보 (bbox 포함)
            all_tracks: 모든 트랙 리스트 (겹침 감지용)
            
        Returns:
            tuple: (cropped_image, feature_vector) 또는 (None, None) (겹침 시)
        """
        bbox = track["bbox"]
        
        # 겹침 감지 (all_tracks가 제공된 경우에만)
        if all_tracks is not None:
            # 겹침 감지
            if self.detect_overlap(bbox, all_tracks, track["track_id"]):
                # print(f"[ImageProcessor] Overlap detected: Track {current_track_id} overlaps with another track")
                # print(f"[ImageProcessor] Skipping crop for Track {current_track_id} due to overlap")
                return None, None
        
        crop = self.crop_bbox_from_frame(frame, bbox)
        feature = self.extract_feature(crop)
        return crop, feature
    
    def get_bbox_coordinates(self, bbox: list) -> tuple:
        """
        바운딩 박스에서 좌표 추출 (발 위치 기준)
        
        Args:
            bbox: 바운딩 박스 [x1, y1, x2, y2]
            
        Returns:
            tuple: (x1, y1, x2, y2, center_x, center_y)
        """
        x1, y1, x2, y2 = map(int, bbox)
        center_x = (x1 + x2) / 2  # 가로 중앙
        center_y = y2              # 하단 (발 위치) - 수정됨
        return x1, y1, x2, y2, center_x, center_y
