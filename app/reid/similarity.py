from scipy.spatial.distance import cdist
import numpy as np
from typing import Dict, Optional


class FeatureSimilarityCalculator:
    """
    feature 벡터 간의 cosine 유사도를 기반으로 가장 유사한 글로벌 ID를 계산하는 유틸리티
    """

    def find_best_match(self,
                        feature: np.ndarray,
                        candidates: Dict[int, np.ndarray],
                        threshold: float) -> Optional[int]:
        """
        주어진 feature에 대해 후보들 중 가장 유사한 글로벌 ID 반환

        :param feature: 입력 feature (1D 벡터)
        :param candidates: {global_id: feature_vector} 딕셔너리
        :param threshold: cosine 유사도 threshold (낮을수록 더 유사)
        :return: 가장 유사한 글로벌 ID 또는 None
        """
        if not candidates:
            return None

        ids = list(candidates.keys())
        vectors = np.array(list(candidates.values()))

        dists = cdist([feature], vectors, metric='cosine')[0]
        min_index = np.argmin(dists)
        min_dist = dists[min_index]

        if min_dist < threshold:
            return ids[min_index]
        return None

    def calculate_similarity(self, feature1: np.ndarray, feature2: np.ndarray) -> float:
        """
        두 feature 벡터 간의 cosine 유사도 계산
        
        :param feature1: 첫 번째 feature 벡터
        :param feature2: 두 번째 feature 벡터
        :return: cosine 유사도 (0~1, 높을수록 유사)
        """
        # 정규화
        feature1_norm = feature1 / np.linalg.norm(feature1)
        feature2_norm = feature2 / np.linalg.norm(feature2)
        
        # cosine 유사도 계산
        similarity = 1 - cdist([feature1_norm], [feature2_norm], metric='cosine')[0][0]
        return similarity
