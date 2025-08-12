import redis
import pickle
import threading
import numpy as np
from typing import List, Dict, Optional
from config import settings


class FeatureStoreRedisHandler:
    """
    Redis에 feature 데이터를 저장, 조회, 삭제하는 기능을 담당.
    카메라/트랙 ID별 관리 및 TTL 설정도 포함.
    원본의 복잡한 기능을 지원: 메타데이터 저장, 카메라별 조회, 사라진 객체 관리
    """

    def __init__(self, redis_host=None, redis_port=None, feature_ttl=None):
        # 글로벌 Redis 연결 설정 사용
        if redis_host is None:
            redis_host = settings.REDIS_CONFIG["host"]
        if redis_port is None:
            redis_port = settings.REDIS_CONFIG["port"]
            
        # ReID 전용 TTL 설정 사용
        if feature_ttl is None:
            feature_ttl = settings.REID_CONFIG["redis"]["feature_ttl"]
            
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
        self.feature_ttl = feature_ttl
        self.lock = threading.Lock()
        self.global_id_counter_key = "global_track_id_counter"
        
        # ReID 전용 Redis 설정 가져오기
        self.track_ttl = settings.REID_CONFIG["redis"]["track_ttl"]
        self.max_features_per_track = settings.REID_CONFIG["redis"]["max_features_per_track"]
        self.pre_registration_ttl = settings.REID_CONFIG["redis"]["pre_registration_ttl"]
        self.cleanup_buffer = settings.REID_CONFIG["redis"]["cleanup_buffer"]

    def _make_track_key(self, global_id: int, camera_id: str, local_track_id: int) -> str:
        return f"global_track:{global_id}:{camera_id}:{local_track_id}"

    def _make_track_data_key(self, global_id: int, camera_id: Optional[str] = None, local_track_id: Optional[int] = None) -> str:
        """
        신규 스키마 키 생성자.
        - 선호: global_track_data:{global_id}:{camera_id}:{local_track_id}
        - (이전 레거시) global_track_data:{global_id}
        """
        if camera_id is not None and local_track_id is not None:
            return f"global_track_data:{global_id}:{camera_id}:{local_track_id}"
        # 레거시: 주석 처리 대상이지만 하위호환 호출을 위해 남겨둠
        return f"global_track_data:{global_id}"

    def store_feature(self, global_id: int, camera_id: str, local_track_id: int, feature: np.ndarray):
        """기본 feature 저장 (하위 호환성)"""
        key = self._make_track_key(global_id, camera_id, local_track_id)
        data = pickle.dumps(feature)
        with self.lock:
            self.redis.setex(key, self.feature_ttl, data)

    def store_feature_with_metadata(self, global_id: int, camera_id: str, frame_id: int,
                                  feature: np.ndarray, bbox: List[int],
                                  global_frame_counter: int,
                                  local_track_id: Optional[int] = None):
        """메타데이터와 함께 feature 저장 (신규 per-cam-local 키만 사용)"""

        camera_id_str = str(camera_id)
        # 신규 스키마 키를 기존 변수명인 data_key로 사용
        data_key = self._make_track_data_key(
            global_id, camera_id_str, local_track_id
        )

        with self.lock:
            track_data = self.redis.get(data_key)
            if track_data:
                track_info = pickle.loads(track_data)
                if not isinstance(track_info, dict):
                    track_info = {'features': [], 'last_seen': 0, 'last_bbox': [0, 0, 0, 0]}
            else:
                track_info = {'features': [], 'last_seen': 0, 'last_bbox': [0, 0, 0, 0]}

            if feature is not None:
                track_info.setdefault('features', []).append(feature)
                # feature 개수 제한 (설정값 사용)
                if len(track_info['features']) > self.max_features_per_track:
                    # 가장 오래된 feature부터 제거 (FIFO)
                    track_info['features'] = track_info['features'][-self.max_features_per_track:]
            track_info['last_seen'] = global_frame_counter
            track_info['last_bbox'] = bbox

            self.redis.set(data_key, pickle.dumps(track_info))

    def create_new_track(self, global_id: int, camera_id: str, frame_id: int,
                        feature: np.ndarray, bbox: List[int], global_frame_counter: int,
                        local_track_id: Optional[int] = None):
        """새로운 트랙 생성 (신규 per-cam-local 키만 사용)"""

        camera_id_str = str(camera_id)
        data_key = self._make_track_data_key(
            global_id, camera_id_str, local_track_id
        )

        track_info = {
            'features': [feature] if feature is not None else [],
            'last_seen': global_frame_counter,
            'last_bbox': bbox
        }

        with self.lock:
            self.redis.set(data_key, pickle.dumps(track_info))
            print(f"Redis: Created new track {global_id} for camera {camera_id}")

    def create_pre_registered_track(self, global_id: int, camera_id: str, frame_id: int,
                                   feature: np.ndarray, bbox: List[int], 
                                   local_track_id: Optional[int] = None):
        """
        사전 등록된 Global ID로 새로운 track 생성
        다른 track들과 동일한 구조로 생성하여 통합 관리
        """
        try:
            camera_id_str = str(camera_id)
            data_key = self._make_track_data_key(
                global_id, camera_id_str, local_track_id
            )

            track_info = {
                'features': [feature] if feature is not None else [],
                'last_seen': frame_id,
                'last_bbox': bbox
            }

            with self.lock:
                self.redis.set(data_key, pickle.dumps(track_info))
                print(f"Redis: Created pre-registered track {global_id} for camera {camera_id}, local_track {local_track_id}")
                
        except Exception as e:
            print(f"Redis: Failed to create pre-registered track: {str(e)}")

    def get_candidate_features(self, exclude_camera: str = None) -> Dict[int, np.ndarray]:
        """기본 candidate features 조회 (하위 호환성)"""
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

    def get_candidate_features_by_camera(self, camera_id: str) -> Dict[int, Dict]:
        """특정 카메라의 candidate features 조회 (신규 per-cam-local 키 형식 사용)"""
        result: Dict[int, Dict] = {}
        # 신규 스키마 키 패턴만 조회
        data_keys = self.redis.keys("global_track_data:*:*:*")

        print(f"Redis: Searching for candidates in camera {camera_id}, found {len(data_keys)} data keys")

        for raw_key in data_keys:
            try:
                # 키 파싱
                data_key = raw_key  # 기존 변수명 재사용
                key_parts = data_key.decode().split(":")
                # global_track_data:{global_id}:{camera_id}:{local_track_id}
                if len(key_parts) != 4:
                    continue

                global_id = int(key_parts[1])
                camera_id_str = str(camera_id)
                key_camera = key_parts[2]
                if key_camera != camera_id_str:
                    continue

                # 값 로드
                track_data = self.redis.get(data_key)
                if not track_data:
                    continue
                track_info = pickle.loads(track_data)
                if not isinstance(track_info, dict):
                    continue

                # 병합: 동일 global_id의 여러 local_track_id를 하나로
                entry = result.get(global_id, {'features': [], 'bbox': [0, 0, 0, 0], 'last_seen': 0})
                features = track_info.get('features', [])
                if isinstance(features, list):
                    entry['features'].extend(features)
                last_seen = int(track_info.get('last_seen', 0))
                if last_seen >= entry.get('last_seen', 0):
                    entry['last_seen'] = last_seen
                    entry['bbox'] = track_info.get('last_bbox', entry['bbox'])
                result[global_id] = entry

                print(f"Redis: Found candidate track {global_id} for camera {camera_id}")
            except Exception as e:
                print(f"Redis: Error processing data key {data_key}: {e}")
                continue

        print(f"Redis: Returning {len(result)} candidates for camera {camera_id}")
        return result

    def cleanup_expired_tracks(self, global_frame_counter: int, ttl_frames: int):
        """만료된 트랙 정리 (신규 per-cam-local 키 기준)"""
        data_keys = self.redis.keys("global_track_data:*:*:*")

        # global_id별 최대 last_seen 집계
        max_seen_by_global: Dict[int, int] = {}
        for k in data_keys:
            try:
                parts = k.decode().split(":")
                if len(parts) != 4:
                    continue
                global_id = int(parts[1])
                val = self.redis.get(k)
                if not val:
                    continue
                info = pickle.loads(val)
                if not isinstance(info, dict):
                    continue
                last_seen = int(info.get('last_seen', 0))
                prev = max_seen_by_global.get(global_id, 0)
                if last_seen > prev:
                    max_seen_by_global[global_id] = last_seen
            except Exception:
                continue

        # 활성 객체는 TTL * cleanup_buffer, 사라진 상태 플래그는 신규 스키마에 없으므로 모두 활성로 간주
        for gid, max_last_seen in max_seen_by_global.items():
            ttl = ttl_frames * self.cleanup_buffer
            if global_frame_counter - max_last_seen > ttl:
                self._remove_track(gid)
                print(f"Redis: Expired track {gid}")

    def _remove_track(self, global_id: int):
        """트랙 완전 제거 (신규 per-cam-local 키만 삭제)"""

        new_data_keys = self.redis.keys(f"global_track_data:{global_id}:*:*")
        history_keys = self.redis.keys(f"track_history:*:{global_id}")
        keys_to_delete = list(new_data_keys) + list(history_keys)
        if keys_to_delete:
            self.redis.delete(*keys_to_delete)

    def generate_new_global_id(self) -> int:
        return self.redis.incr(self.global_id_counter_key)
