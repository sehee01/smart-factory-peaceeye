import cv2
import numpy as np
import torch
from torchreid.utils.feature_extractor import FeatureExtractor
from config import settings


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
    
    def process_track_for_reid(self, frame: np.ndarray, track: dict) -> tuple:
        """
        트랙 정보를 받아서 크롭 및 feature 추출을 수행
        
        Args:
            frame: 입력 프레임
            track: 트랙 정보 (bbox 포함)
            
        Returns:
            tuple: (cropped_image, feature_vector)
        """
        bbox = track["bbox"]
        crop = self.crop_bbox_from_frame(frame, bbox)
        feature = self.extract_feature(crop)
        return crop, feature
