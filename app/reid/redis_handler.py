import redis
import pickle
import threading
import numpy as np
from typing import List, Dict, Optional


class FeatureStoreRedisHandler:
    """
    Redis에 feature 데이터를 저장, 조회, 삭제하는 기능을 담당.
    카메라/트랙 ID별 관리 및 TTL 설정도 포함.
    원본의 복잡한 기능을 지원: 메타데이터 저장, 카메라별 조회, 사라진 객체 관리
    """

    def __init__(self, redis_host='localhost', redis_port=6379, feature_ttl=300):
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
        self.feature_ttl = feature_ttl
        self.lock = threading.Lock()
        self.global_id_counter_key = "global_track_id_counter"

    def _make_track_key(self, global_id: int, camera_id: str, local_track_id: int) -> str:
        return f"global_track:{global_id}:{camera_id}:{local_track_id}"

    def _make_track_data_key(self, global_id: int) -> str:
        return f"global_track_data:{global_id}"

    def store_feature(self, global_id: int, camera_id: str, local_track_id: int, feature: np.ndarray):
        """기본 feature 저장 (하위 호환성)"""
        key = self._make_track_key(global_id, camera_id, local_track_id)
        data = pickle.dumps(feature)
        with self.lock:
            self.redis.setex(key, self.feature_ttl, data)

    def store_feature_with_metadata(self, global_id: int, camera_id: str, frame_id: int, 
                                  feature: np.ndarray, bbox: List[int], max_features: int, 
                                  global_frame_counter: int):
        """메타데이터와 함께 feature 저장"""
        data_key = self._make_track_data_key(global_id)
        track_key = f"global_track:{global_id}"
        
        with self.lock:
            # 기존 데이터 가져오기
            track_data = self.redis.get(data_key)
            if track_data:
                track_info = pickle.loads(track_data)
            else:
                track_info = {
                    'cameras': {},
                    'is_disappeared': False,
                    'disappeared_since': None,
                    'last_activity': global_frame_counter
                }
            
            camera_id_str = str(camera_id)
            
            # 카메라 정보 초기화 또는 업데이트
            if camera_id_str not in track_info['cameras']:
                track_info['cameras'][camera_id_str] = {
                    'features': [],
                    'last_seen': global_frame_counter,
                    'last_bbox': bbox
                }
            
            # 특징 추가
            if feature is not None:
                track_info['cameras'][camera_id_str]['features'].append(feature)
                
                # 슬라이딩 윈도우 적용
                if len(track_info['cameras'][camera_id_str]['features']) > max_features:
                    track_info['cameras'][camera_id_str]['features'] = \
                        track_info['cameras'][camera_id_str]['features'][-max_features:]
            
            # 메타데이터 업데이트
            track_info['cameras'][camera_id_str]['last_seen'] = global_frame_counter
            track_info['cameras'][camera_id_str]['last_bbox'] = bbox
            track_info['last_activity'] = global_frame_counter
            
            # 사라진 객체가 다시 나타난 경우 상태 복구
            if track_info.get('is_disappeared', False) and feature is not None:
                track_info['is_disappeared'] = False
                track_info['disappeared_since'] = None
                print(f"Redis: Restored disappeared track {global_id} to normal state")
            
            # 저장
            self.redis.set(data_key, pickle.dumps(track_info))
            self.redis.set(track_key, b'1')

    def create_new_track(self, global_id: int, camera_id: str, frame_id: int, 
                        feature: np.ndarray, bbox: List[int], global_frame_counter: int):
        """새로운 트랙 생성"""
        track_key = f"global_track:{global_id}"
        data_key = self._make_track_data_key(global_id)
        
        track_info = {
            'cameras': {
                str(camera_id): {
                    'features': [feature] if feature is not None else [],
                    'last_seen': global_frame_counter,
                    'last_bbox': bbox
                }
            },
            'is_disappeared': False,
            'disappeared_since': None,
            'last_activity': global_frame_counter
        }
        
        with self.lock:
            self.redis.set(data_key, pickle.dumps(track_info))
            self.redis.set(track_key, b'1')

    def create_disappeared_track(self, global_id: int, bbox: List[int], camera_id: str, frame_id: int):
        """사라진 객체용 새로운 트랙 생성"""
        track_key = f"global_track:{global_id}"
        data_key = self._make_track_data_key(global_id)
        
        track_info = {
            'cameras': {
                str(camera_id): {
                    'features': [],
                    'last_seen': frame_id,
                    'last_bbox': bbox
                }
            },
            'is_disappeared': True,
            'disappeared_since': frame_id,
            'last_activity': frame_id
        }
        
        with self.lock:
            # 사라진 객체: TTL 적용
            self.redis.setex(data_key, self.feature_ttl, pickle.dumps(track_info))
            self.redis.setex(track_key, self.feature_ttl, b'1')

    def mark_track_as_disappeared(self, global_id: int, bbox: List[int], camera_id: str, frame_id: int):
        """트랙을 사라진 상태로 표시"""
        data_key = self._make_track_data_key(global_id)
        
        with self.lock:
            track_data = self.redis.get(data_key)
            if track_data:
                track_info = pickle.loads(track_data)
            else:
                track_info = {
                    'cameras': {},
                    'is_disappeared': False,
                    'disappeared_since': None,
                    'last_activity': frame_id
                }
            
            camera_id_str = str(camera_id)
            
            if camera_id_str not in track_info['cameras']:
                track_info['cameras'][camera_id_str] = {
                    'features': [],
                    'last_seen': frame_id,
                    'last_bbox': bbox
                }
            
            # 사라진 상태로 표시
            track_info['is_disappeared'] = True
            track_info['disappeared_since'] = frame_id
            track_info['last_activity'] = frame_id
            track_info['cameras'][camera_id_str]['last_seen'] = frame_id
            track_info['cameras'][camera_id_str]['last_bbox'] = bbox
            
            # 사라진 객체: TTL 적용
            self.redis.setex(data_key, self.feature_ttl, pickle.dumps(track_info))
            track_key = f"global_track:{global_id}"
            self.redis.setex(track_key, self.feature_ttl, b'1')

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
        """특정 카메라의 candidate features 조회"""
        result = {}
        data_keys = self.redis.keys("global_track_data:*")
        
        for data_key in data_keys:
            try:
                global_id = int(data_key.decode().split(":")[2])
                track_data = self.redis.get(data_key)
                
                if track_data:
                    track_info = pickle.loads(track_data)
                    camera_id_str = str(camera_id)
                    
                    if camera_id_str in track_info['cameras']:
                        camera_data = track_info['cameras'][camera_id_str]
                        result[global_id] = {
                            'features': camera_data.get('features', []),
                            'bbox': camera_data.get('last_bbox', [0, 0, 0, 0]),
                            'last_seen': camera_data.get('last_seen', 0)
                        }
            except Exception as e:
                continue
        
        return result

    def cleanup_expired_tracks(self, global_frame_counter: int, ttl_frames: int):
        """만료된 트랙 정리"""
        data_keys = self.redis.keys("global_track_data:*")
        
        for data_key in data_keys:
            try:
                global_id = int(data_key.decode().split(":")[2])
                track_data = self.redis.get(data_key)
                
                if track_data:
                    track_info = pickle.loads(track_data)
                    
                    # 모든 카메라의 마지막 업데이트 시간 확인
                    max_last_seen = 0
                    for camera_data in track_info['cameras'].values():
                        max_last_seen = max(max_last_seen, camera_data.get('last_seen', 0))
                    
                    # TTL 결정
                    is_disappeared = track_info.get('is_disappeared', False)
                    if is_disappeared:
                        ttl = ttl_frames  # 사라진 객체: 기본 TTL
                    else:
                        ttl = ttl_frames * 2  # 활성 객체: 2배 TTL
                    
                    # TTL이 만료된 트랙 제거
                    if global_frame_counter - max_last_seen > ttl:
                        self._remove_track(global_id)
                        status = "disappeared" if is_disappeared else "normal"
                        print(f"Redis: Expired {status} track {global_id}")
            except Exception as e:
                continue

    def _remove_track(self, global_id: int):
        """트랙 완전 제거"""
        track_key = f"global_track:{global_id}"
        data_key = self._make_track_data_key(global_id)
        
        # 트랙 히스토리도 함께 제거
        history_keys = self.redis.keys(f"track_history:*:{global_id}")
        all_keys = [track_key, data_key] + history_keys
        
        self.redis.delete(*all_keys)

    def generate_new_global_id(self) -> int:
        return self.redis.incr(self.global_id_counter_key)
