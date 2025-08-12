import numpy as np
from typing import Optional, Set, List
from ..redis_handler import FeatureStoreRedisHandler
from ..similarity import FeatureSimilarityCalculator


class CrossCameraMatcher:
    """
    다른 카메라와의 객체 매칭을 담당하는 클래스
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
        다른 카메라와 매칭 수행
        
        Args:
            features: 현재 객체의 feature 벡터
            bbox: 바운딩 박스 [x1, y1, x2, y2]
            camera_id: 카메라 ID
            frame_id: 현재 프레임 ID
            matched_tracks: 이미 매칭된 트랙 ID들의 집합
            
        Returns:
            (global_id, similarity) 튜플 또는 None
        """
        candidates = self.redis.get_candidate_features(exclude_camera=camera_id)
        
        best_match_id = None
        best_similarity = 0
        
        for global_id, candidate_features in candidates.items():
            if global_id in matched_tracks:
                continue
            
            if len(candidate_features) > 0:
                # 가중 평균 특징 계산
                features_array = np.array(candidate_features)
                if len(features_array) == 1:
                    weighted_average = features_array[0]
                else:
                    weights = np.linspace(0.5, 1.0, len(features_array))
                    weights = weights / np.sum(weights)
                    weighted_average = np.average(features_array, axis=0, weights=weights)
                
                similarity = self.similarity.calculate_similarity(features, weighted_average)
                
                # 다른 카메라는 더 엄격한 임계값 사용 (더 관대하게)
                if similarity > best_similarity and similarity > self.threshold * 1.0:
                    best_similarity = similarity
                    best_match_id = global_id
        
        if best_match_id:
            print(f"[CrossCameraMatcher] Other camera match: Track {best_match_id} (similarity: {best_similarity:.3f})")
            return best_match_id, best_similarity
        return None
