import numpy as np
from typing import Optional, Set, Dict, Any, List
from ..redis_handler import FeatureStoreRedisHandler
from ..similarity import FeatureSimilarityCalculator
import sys
import os

# app 디렉토리 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(os.path.dirname(current_dir))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from config import settings


class SameCameraMatcher:
    """
    같은 카메라 내에서 객체 매칭을 담당하는 클래스
    """
    
    def __init__(self, redis_handler: FeatureStoreRedisHandler, 
                 similarity_calc: FeatureSimilarityCalculator,
                 similarity_threshold: float = 0.7):
        self.redis = redis_handler
        self.similarity = similarity_calc
        self.threshold = similarity_threshold
    
    def match(self, features: np.ndarray, bbox: List[int], camera_id: str, 
              frame_id: int, matched_tracks: Set[int]) -> Optional[tuple]:
        """
        같은 카메라 내에서 매칭 수행
        
        Args:
            features: 현재 객체의 feature 벡터
            bbox: 바운딩 박스 [x1, y1, x2, y2]
            camera_id: 카메라 ID
            frame_id: 현재 프레임 ID
            matched_tracks: 이미 매칭된 트랙 ID들의 집합
            
        Returns:
            (global_id, similarity) 튜플 또는 None
        """
        candidates = self.redis.get_candidate_features_by_camera(camera_id)
        
        print(f"[SameCameraMatcher] Same camera matching for camera {camera_id}: found {len(candidates)} candidates")
        
        best_match_id = None
        best_similarity = 0
        
        for global_id, candidate_data in candidates.items():
            if global_id in matched_tracks:
                continue
            
            candidate_features = candidate_data['features']
            candidate_bbox = candidate_data.get('bbox', bbox)
            
            # 위치 기반 필터링 (같은 카메라에서만) - 더 관대하게
            location_score = self._calculate_location_score(bbox, candidate_bbox)
            if location_score < 0.05:  # 더 관대하게 (0.1 -> 0.05)
                continue #location_score가 현재 0또는 1이여서 BBOX거리100픽셀 이상이면 매칭 안됨
            
            # 특징 유사도 계산
            if len(candidate_features) > 0:
                # 가중 평균 특징 계산
                features_array = np.array(candidate_features)
                if len(features_array) == 1:
                    weighted_average = features_array[0]
                else:
                    weights = np.linspace(0.5, 1.0, len(features_array))
                    weights = weights / np.sum(weights)
                    weighted_average = np.average(features_array, axis=0, weights=weights)
                
                feature_similarity = self.similarity.calculate_similarity(features, weighted_average)
                
                # 위치 기반 동적 임계값 계산 (더 관대하게)
                # 위치가 가까우면 임계값을 낮춤 (더 관대한 매칭)
                dynamic_threshold = self.threshold * (1.0 - location_score * 0.7)
                # 최소 임계값 보장 (더 낮게)
                dynamic_threshold = max(dynamic_threshold, self.threshold * 0.2)
                
                print(f"[SameCameraMatcher] Track {global_id}: similarity={feature_similarity:.3f}, threshold={dynamic_threshold:.3f}, location_score={location_score:.3f}")
                
                if feature_similarity > best_similarity and feature_similarity > dynamic_threshold:
                    best_similarity = feature_similarity
                    best_match_id = global_id
                    print(f"[SameCameraMatcher] New best match: Track {global_id} (similarity: {feature_similarity:.3f})")
        
        if best_match_id:
            print(f"[SameCameraMatcher] Same camera match: Track {best_match_id} (similarity: {best_similarity:.3f})")
            return best_match_id, best_similarity
        return None
    
    def _calculate_location_score(self, bbox1: List[int], bbox2: List[int]) -> float:
        """두 바운딩 박스 간의 위치 유사도 계산"""
        center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
        center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]
        
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # 거리 기반 위치 점수 계산 (가까울수록 높은 점수)
        max_distance = 100
        if distance <= max_distance:
            return 1.0 - (distance / max_distance)  # 0~1 사이 점수
        else:
            return 0.0
