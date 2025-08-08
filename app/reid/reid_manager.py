from typing import List
from redis_handler import FeatureStoreRedisHandler
from similarity import FeatureSimilarityCalculator
import numpy as np


class GlobalReIDManager:
    """
    객체의 appearance feature를 기반으로 Redis에 저장된 트랙들과 유사도를 계산하여
    글로벌 ID를 재부여하는 책임 클래스.
    """

    def __init__(self,
                 redis_handler: FeatureStoreRedisHandler,
                 similarity_calc: FeatureSimilarityCalculator,
                 similarity_threshold: float = 0.7):
        self.redis = redis_handler
        self.similarity = similarity_calc
        self.threshold = similarity_threshold

    def reassign_global_id(self, camera_id: str, local_track_id: int, feature: np.ndarray) -> int:
        """
        현재 객체의 feature를 기반으로 Redis에 있는 다른 트랙들과 비교하여
        가장 유사한 글로벌 ID를 할당함

        :param camera_id: 현재 카메라 ID
        :param local_track_id: 해당 카메라 내에서의 트랙 ID
        :param feature: 현재 객체의 appearance feature (1D np.array)
        :return: 재할당된 global track ID (int)
        """
        # 1. 후보 목록 조회 (자신 제외)
        candidates = self.redis.get_candidate_features(exclude_camera=camera_id)

        # 2. 유사도 비교
        matched_id = self.similarity.find_best_match(feature, candidates, self.threshold)

        # 3. ID 결정 및 Redis 업데이트
        if matched_id:
            global_id = matched_id
        else:
            global_id = self.redis.generate_new_global_id()

        self.redis.store_feature(global_id, camera_id, local_track_id, feature)
        return global_id
