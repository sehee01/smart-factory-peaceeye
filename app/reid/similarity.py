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
