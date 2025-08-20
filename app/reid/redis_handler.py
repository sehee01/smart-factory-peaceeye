import redis
import pickle
import threading
import numpy as np
from typing import List, Dict, Optional
from app.config import settings


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
        """

        return f"global_track_data:{global_id}:{camera_id}:{local_track_id}"
    

    def store_feature(self, global_id: int, camera_id: str, local_track_id: int, feature: np.ndarray):
        """기본 feature 저장"""
        key = self._make_track_key(global_id, camera_id, local_track_id)
        data = pickle.dumps(feature)
        with self.lock:
            self.redis.setex(key, self.feature_ttl, data)

    def store_feature_with_metadata(self, global_id: int, camera_id: str, frame_id: int,
                                  feature: np.ndarray, bbox: List[int],
                                  global_frame_counter: int,
                                  local_track_id: Optional[int] = None):
        """메타데이터와 함께 feature 저장 """

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
                    track_info = {'features': [], 'last_seen': 0, 'last_bbox': [0, 0, 0, 0], 'is_tracking': True}
            else:
                track_info = {'features': [], 'last_seen': 0, 'last_bbox': [0, 0, 0, 0], 'is_tracking': True}

            if feature is not None:
                track_info.setdefault('features', []).append(feature)
                # feature 개수 제한 (설정값 사용)
                if len(track_info['features']) > self.max_features_per_track:
                    # 가장 오래된 feature부터 제거 (FIFO)
                    track_info['features'] = track_info['features'][-self.max_features_per_track:]
            
            # 객체가 감지되었으므로 트래킹 중 상태로 설정
            track_info['last_seen'] = global_frame_counter
            track_info['last_bbox'] = bbox
            track_info['is_tracking'] = True  # 현재 트래킹 중

            # 트래킹 중인 객체는 TTL 무제한 (Redis에서 -1은 무제한)
            self.redis.set(data_key, pickle.dumps(track_info))
            # print(f"Redis: Track {global_id} is actively tracking (TTL: unlimited)")

    def mark_track_as_failed(self, global_id: int, camera_id: str, local_track_id: int, global_frame_counter: int):
        """트랙을 실패 상태로 표시 (트래킹에 실패한 객체)"""
        camera_id_str = str(camera_id)
        data_key = self._make_track_data_key(global_id, camera_id_str, local_track_id)
        
        with self.lock:
            track_data = self.redis.get(data_key)
            if track_data:
                track_info = pickle.loads(track_data)
                if isinstance(track_info, dict):
                    track_info['is_tracking'] = False  # 트래킹 실패 상태로 표시
                    track_info['failed_at'] = global_frame_counter  # 실패 시점 기록
                    
                    # 트래킹에 실패한 객체는 TTL 10초 설정
                    self.redis.setex(data_key, 10, pickle.dumps(track_info))
                    # print(f"Redis: Marked track {global_id} as failed (TTL: 10 seconds)")

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
            'last_bbox': bbox,
            'is_tracking': True  # 새로 생성된 트랙은 트래킹 중 상태
        }

        with self.lock:
            # 트래킹 중인 객체는 TTL 무제한
            self.redis.set(data_key, pickle.dumps(track_info))
            # print(f"Redis: Created new track {global_id} for camera {camera_id} (TTL: unlimited)")

    def create_pre_registered_track(self, global_id: int, camera_id: str, frame_id: int,
                                   feature: np.ndarray, bbox: List[int], 
                                   local_track_id: Optional[int] = None):
        """
        사전 등록된 Global ID로 새로운 track 생성
        일반 트랙과 동일한 데이터 구조로 생성하여 통합 관리
        """
        try:
            camera_id_str = str(camera_id)
            data_key = self._make_track_data_key(
                global_id, camera_id_str, local_track_id
            )

            # 일반 트랙과 동일한 데이터 구조
            track_info = {
                'features': [feature] if feature is not None else [],
                'last_seen': frame_id,
                'last_bbox': bbox,
                'is_tracking': True  # 사전 등록된 트랙도 트래킹 중 상태
            }

            with self.lock:
                # 트래킹 중인 객체는 TTL 무제한
                self.redis.set(data_key, pickle.dumps(track_info))
                # print(f"Redis: Created pre-registered track {global_id} for camera {camera_id}, local_track {local_track_id} (TTL: unlimited)")
                
        except Exception as e:
            print(f"Redis: Failed to create pre-registered track: {str(e)}")

    def get_candidate_features(self, exclude_camera: str = None) -> Dict[int, Dict]:
        result = {}
        
        if exclude_camera:
            # exclude_camera가 아닌 다른 카메라들만 조회
            all_keys = self.redis.keys("global_track_data:*:*:*")
            
            for raw_key in all_keys:
                try:
                    key_parts = raw_key.decode().split(":")
                    if len(key_parts) != 4:
                        continue
                        
                    global_id = int(key_parts[1])
                    camera_id = key_parts[2]
                    
                    # exclude_camera 조건 확인
                    if camera_id == exclude_camera:
                        continue
                    
                    # 값 로드 및 처리
                    track_data = self.redis.get(raw_key)
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
                        
                except Exception as e:
                    continue
        
        return result

    def get_candidate_features_by_camera(self, camera_id: str) -> Dict[int, Dict]:
        """특정 카메라의 candidate features 조회 (신규 per-cam-local 키 형식 사용)"""
        result: Dict[int, Dict] = {}
        
        # 특정 카메라의 키만 조회 (최적화)
        pattern = f"global_track_data:*:{camera_id}:*"
        data_keys = self.redis.keys(pattern)

        for raw_key in data_keys:
            try:
                # 키 파싱
                data_key = raw_key  # 기존 변수명 재사용
                key_parts = data_key.decode().split(":")
                # global_track_data:{global_id}:{camera_id}:{local_track_id}
                if len(key_parts) != 4:
                    continue

                global_id = int(key_parts[1])
                local_track_id = int(key_parts[3])  # local_track_id 추출
                
                # 값 로드
                track_data = self.redis.get(data_key)
                if not track_data:
                    continue
                track_info = pickle.loads(track_data)
                if not isinstance(track_info, dict):
                    continue

                # 기존 entry가 있고, 현재 local_track_id가 더 작으면 건너뛰기
                if global_id in result:
                    existing_local_id = int(result[global_id].get('local_track_id', 0))
                    if local_track_id <= existing_local_id:
                        continue

                # 새로운 entry 생성 (기존 것 대체)
                features = track_info.get('features', [])
                entry = {
                    'features': features if isinstance(features, list) else [],
                    'bbox': track_info.get('last_bbox', [0, 0, 0, 0]),
                    'last_seen': int(track_info.get('last_seen', 0)),
                    'local_track_id': local_track_id  # local_track_id 저장
                }
                result[global_id] = entry
            except Exception as e:
                print(f"Redis: Error processing data key {data_key}: {e}")
                continue

        return result

    def cleanup_expired_tracks(self, global_frame_counter: int, ttl_frames: int):
        """
        만료된 트랙들을 정리
        """
        try:
            # 모든 트랙 데이터 키 조회
            pattern = "global_track_data:*"
            keys = self.redis.keys(pattern)
            
            deleted_count = 0
            for raw_key in keys:
                try:
                    key_parts = raw_key.decode().split(":")
                    if len(key_parts) == 4:
                        global_id = int(key_parts[1])
                        camera_id = key_parts[2]
                        local_track_id = int(key_parts[3])
                        
                        # 트랙 데이터 로드
                        track_data = self.redis.get(raw_key)
                        if track_data:
                            track_info = pickle.loads(track_data)
                            last_seen = track_info.get('last_seen', 0)
                            
                            # 만료 체크
                            if global_frame_counter - last_seen > ttl_frames:
                                self.redis.delete(raw_key)
                                deleted_count += 1
                                # print(f"Redis: Expired track {gid} (default TTL)")
                                
                except Exception as e:
                    print(f"Redis: Error cleaning up track {raw_key}: {e}")
                    continue
            
            if deleted_count > 0:
                print(f"Redis: Cleaned up {deleted_count} expired tracks")
                
        except Exception as e:
            print(f"Redis: Error in cleanup_expired_tracks: {e}")

    def generate_new_global_id(self) -> int:
        return self.redis.incr(self.global_id_counter_key)
