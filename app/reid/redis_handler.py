import redis
import pickle
import threading
import numpy as np
from typing import List, Dict


class FeatureStoreRedisHandler:
    """
    Redis에 feature 데이터를 저장, 조회, 삭제하는 기능을 담당.
    카메라/트랙 ID별 관리 및 TTL 설정도 포함.
    """

    def __init__(self, redis_host='localhost', redis_port=6379, feature_ttl=300):
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
        self.feature_ttl = feature_ttl
        self.lock = threading.Lock()
        self.global_id_counter_key = "global_track_id_counter"

    def _make_track_key(self, global_id: int, camera_id: str, local_track_id: int) -> str:
        return f"global_track:{global_id}:{camera_id}:{local_track_id}"

    def store_feature(self, global_id: int, camera_id: str, local_track_id: int, feature: np.ndarray):
        key = self._make_track_key(global_id, camera_id, local_track_id)
        data = pickle.dumps(feature)
        with self.lock:
            self.redis.setex(key, self.feature_ttl, data)

    def get_candidate_features(self, exclude_camera: str = None) -> Dict[int, np.ndarray]:
        result = {}
        keys = self.redis.keys("global_track:*")
        for key in keys:
            try:
                key_parts = key.decode().split(":")  # format: global_track:{id}:{camera_id}:{track_id}
                global_id = int(key_parts[1])
                camera_id = key_parts[2]
                if exclude_camera and camera_id == exclude_camera:
                    continue

                data = self.redis.get(key)
                if data:
                    feature = pickle.loads(data)
                    result[global_id] = feature
            except Exception as e:
                continue  # skip malformed or expired keys
        return result

    def generate_new_global_id(self) -> int:
        return self.redis.incr(self.global_id_counter_key)
