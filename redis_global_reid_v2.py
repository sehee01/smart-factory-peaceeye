import redis
import json
import numpy as np
import pickle
import threading
from scipy.spatial.distance import cdist
from ByteTrack.yolox.tracker.basetrack import BaseTrack

class RedisGlobalReIDManagerV2:
    """
    카메라별 우선순위 매칭과 다중 카메라 정보 통합을 지원하는 Redis Global ReID 매니저
    사라지는 객체 감지 및 TTL 제외 관리 포함
    """
    def __init__(self, similarity_threshold=0.7, feature_ttl=300, max_features_per_camera=10, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
        self.similarity_threshold = similarity_threshold
        self.feature_ttl = feature_ttl
        self.max_features_per_camera = max_features_per_camera
        self.lock = threading.Lock()
        
        # Redis 키 패턴 (통합)
        self.track_key_pattern = "global_track:{}"
        self.track_data_key_pattern = "global_track_data:{}"
        self.track_history_key_pattern = "track_history:{}:{}"  # camera_id:track_id
    
    def update_frame(self, frame_id):
        """현재 프레임 업데이트 및 만료된 트랙 정리 (통합된 구조)"""
        with self.lock:
            track_keys = self.redis_client.keys("global_track:*")
            
            for track_key in track_keys:
                track_id = track_key.decode().split(':')[1]
                data_key = self.track_data_key_pattern.format(track_id)
                
                # 트랙 데이터 가져오기
                track_data = self.redis_client.get(data_key)
                if track_data:
                    track_info = pickle.loads(track_data)
                    
                    # 모든 카메라의 마지막 업데이트 시간 확인
                    max_last_seen = 0
                    for camera_data in track_info['cameras'].values():
                        max_last_seen = max(max_last_seen, camera_data.get('last_seen', 0))
                    
                    # TTL 결정 (사라진 객체는 기본 TTL)
                    is_disappeared = track_info.get('is_disappeared', False)
                    if is_disappeared:
                        ttl = self.feature_ttl  # 사라진 객체는 기본 TTL (5분)
                    else:
                        ttl = self.feature_ttl  # 일반 객체는 기본 TTL (5분)
                    
                    # TTL이 만료된 트랙 제거
                    if frame_id - max_last_seen > ttl:
                        self._remove_track(track_id)
                        status = "disappeared" if is_disappeared else "normal"
                        print(f"Global ReID: Expired {status} track {track_id}")
    
    def _remove_track(self, track_id):
        """트랙 완전 제거 (통합된 구조)"""
        track_key = self.track_key_pattern.format(track_id)
        data_key = self.track_data_key_pattern.format(track_id)
        
        # 트랙 히스토리도 함께 제거
        history_keys = self.redis_client.keys(f"track_history:*:{track_id}")
        all_keys = [track_key, data_key] + history_keys
        
        self.redis_client.delete(*all_keys)
    
    def detect_disappearing_object(self, bbox, frame_shape, previous_bbox_area=None):
        """객체가 화면에서 사라지는지 감지"""
        x1, y1, x2, y2 = bbox
        frame_height, frame_width = frame_shape
        
        # 바운딩 박스가 화면 경계에 닿는지 확인
        touching_boundary = (x1 <= 0 or y1 <= 0 or x2 >= frame_width or y2 >= frame_height)
        
        # 바운딩 박스 크기가 급격히 작아지는지 확인
        bbox_area = (x2 - x1) * (y2 - y1)
        is_shrinking = False
        if previous_bbox_area:
            is_shrinking = bbox_area < previous_bbox_area * 0.5
        
        return touching_boundary or is_shrinking
    
    def get_disappearing_zone(self, bbox, frame_shape):
        """객체가 사라지는 영역을 구분"""
        x1, y1, x2, y2 = bbox
        frame_height, frame_width = frame_shape
        
        # 화면 경계에 닿는 방향 확인
        touching_left = x1 <= 0
        touching_right = x2 >= frame_width
        touching_top = y1 <= 0
        touching_bottom = y2 >= frame_height
        
        if touching_left and touching_top:
            return "left_top"
        elif touching_right and touching_top:
            return "right_top"
        elif touching_left and touching_bottom:
            return "left_bottom"
        elif touching_right and touching_bottom:
            return "right_bottom"
        elif touching_left:
            return "left"
        elif touching_right:
            return "right"
        elif touching_top:
            return "top"
        elif touching_bottom:
            return "bottom"
        else:
            return "center"
    
    def match_or_create(self, features, bbox, camera_id, frame_id, frame_shape, matched_tracks=None):
        """
        사라지는 객체는 ReID 스킵하는 Global ReID 매칭
        
        Args:
            features: ReID 특징 벡터 (numpy array)
            bbox: 바운딩 박스 [x1, y1, x2, y2]
            camera_id: 카메라 ID
            frame_id: 현재 프레임 ID
            frame_shape: 프레임 크기 (height, width)
            matched_tracks: 이미 매칭된 트랙 ID들의 집합
            
        Returns:
            global_id: 매칭된 또는 새로 생성된 글로벌 ID
        """
        if matched_tracks is None:
            matched_tracks = set()
        
        if features is None or len(features) == 0:
            return None
        
        with self.lock:
            # 1단계: 사라지는 객체 감지
            previous_bbox_area = self._get_previous_bbox_area(camera_id, frame_id)
            is_disappearing = self.detect_disappearing_object(bbox, frame_shape, previous_bbox_area)
            
            if is_disappearing:
                print(f"Global ReID: Skipping ReID for disappearing object at {self.get_disappearing_zone(bbox, frame_shape)}")
                # 사라지는 객체는 기존 ID 유지 (ReID 스킵)
                return self._get_existing_id_for_disappearing_object(bbox, camera_id, frame_id, matched_tracks)
            
            # 2단계: 정상적인 ReID 매칭
            # 특징 벡터 정규화
            features = features / np.linalg.norm(features)
            
            best_match_id = None
            best_similarity = 0
            best_match_camera = None
            
            # 같은 카메라 내 매칭 (높은 우선순위)
            same_camera_match = self._match_same_camera(features, bbox, camera_id, frame_id, matched_tracks)
            if same_camera_match:
                best_match_id, best_similarity, best_match_camera = same_camera_match
                print(f"Global ReID: Same camera match - Track {best_match_id} (similarity: {best_similarity:.3f})")
            
            # 다른 카메라 매칭 (낮은 우선순위, 같은 카메라 매칭이 없을 때만)
            if best_match_id is None:
                other_camera_match = self._match_other_cameras(features, bbox, camera_id, frame_id, matched_tracks)
                if other_camera_match:
                    best_match_id, best_similarity, best_match_camera = other_camera_match
                    print(f"Global ReID: Cross camera match - Track {best_match_id} (similarity: {best_similarity:.3f})")
            
            if best_match_id is not None:
                # 기존 트랙에 현재 카메라 정보 추가/업데이트
                self._update_track_camera(best_match_id, features, bbox, camera_id, frame_id)
                matched_tracks.add(best_match_id)
                return best_match_id
            else:
                # 새로운 글로벌 ID 생성
                global_id = BaseTrack.next_id()
                self._create_track(global_id, features, bbox, camera_id, frame_id)
                print(f"Global ReID: Created new track {global_id}")
                return global_id
    
    def _get_previous_bbox_area(self, camera_id, frame_id):
        """이전 프레임의 바운딩 박스 크기 가져오기"""
        # 간단한 구현: 실제로는 더 정교한 히스토리 관리 필요
        return None
    
    def _match_same_camera(self, features, bbox, camera_id, frame_id, matched_tracks):
        """같은 카메라 내 매칭"""
        track_keys = self.redis_client.keys("global_track:*")
        
        best_match_id = None
        best_similarity = 0
        best_match_camera = None
        
        for track_key in track_keys:
            track_id = track_key.decode().split(':')[1]
            
            if track_id in matched_tracks:
                continue
            
            # 해당 카메라의 데이터가 있는지 확인
            data_key = self.track_data_key_pattern.format(track_id)
            track_data = self.redis_client.get(data_key)
            
            if track_data:
                track_info = pickle.loads(track_data)
                
                # 같은 카메라의 데이터가 있는지 확인
                if str(camera_id) in track_info['cameras']:
                    camera_data = track_info['cameras'][str(camera_id)]
                    
                    # 위치 기반 필터링 (같은 카메라에서만)
                    location_score = 0
                    if 'last_bbox' in camera_data:
                        last_bbox = camera_data['last_bbox']
                        current_center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                        last_center = [(last_bbox[0] + last_bbox[2]) / 2, (last_bbox[1] + last_bbox[3]) / 2]
                        distance = np.sqrt((current_center[0] - last_center[0])**2 + (current_center[1] - last_center[1])**2)
                        
                        # 거리 기반 위치 점수 계산 (가까울수록 높은 점수)
                        max_distance = 200
                        if distance <= max_distance:
                            location_score = 1.0 - (distance / max_distance)  # 0~1 사이 점수
                        else:
                            continue  # 너무 멀면 제외
                    else:
                        location_score = 0.5  # 위치 정보가 없는 경우 중간 점수
                    
                    # 특징 유사도 계산
                    camera_features = camera_data['features']
                    if len(camera_features) > 0:
                        # 가중 평균 특징 계산
                        features_array = np.array(camera_features)
                        if len(features_array) == 1:
                            weighted_average = features_array[0]
                        else:
                            weights = np.linspace(0.5, 1.0, len(features_array))
                            weights = weights / np.sum(weights)
                            weighted_average = np.average(features_array, axis=0, weights=weights)
                        
                        feature_similarity = 1 - cdist([features], [weighted_average], 'cosine')[0][0]
                        
                        # 위치 기반 동적 임계값 계산
                        # 위치가 가까우면 임계값을 낮춤 (더 관대한 매칭)
                        dynamic_threshold = self.similarity_threshold * (1.0 - location_score * 0.5)
                        # 최소 임계값 보장 (너무 낮아지지 않도록)
                        dynamic_threshold = max(dynamic_threshold, self.similarity_threshold * 0.3)
                        
                        if feature_similarity > best_similarity and feature_similarity > dynamic_threshold:
                            best_similarity = feature_similarity
                            best_match_id = track_id
                            best_match_camera = camera_id
        
        if best_match_id:
            return best_match_id, best_similarity, best_match_camera
        return None
    
    def _match_other_cameras(self, features, bbox, camera_id, frame_id, matched_tracks):
        """다른 카메라와 매칭"""
        track_keys = self.redis_client.keys("global_track:*")
        
        best_match_id = None
        best_similarity = 0
        best_match_camera = None
        
        for track_key in track_keys:
            track_id = track_key.decode().split(':')[1]
            
            if track_id in matched_tracks:
                continue
            
            # 해당 카메라의 데이터가 있는지 확인
            data_key = self.track_data_key_pattern.format(track_id)
            track_data = self.redis_client.get(data_key)
            
            if track_data:
                track_info = pickle.loads(track_data)
                
                # 모든 카메라의 특징들과 비교
                for cam_id, camera_data in track_info['cameras'].items():
                    if int(cam_id) == camera_id:  # 같은 카메라는 건너뛰기
                        continue
                    
                    camera_features = camera_data['features']
                    if len(camera_features) > 0:
                        # 가중 평균 특징 계산
                        features_array = np.array(camera_features)
                        if len(features_array) == 1:
                            weighted_average = features_array[0]
                        else:
                            weights = np.linspace(0.5, 1.0, len(features_array))
                            weights = weights / np.sum(weights)
                            weighted_average = np.average(features_array, axis=0, weights=weights)
                        
                        similarity = 1 - cdist([features], [weighted_average], 'cosine')[0][0]
                        
                        if similarity > best_similarity and similarity > self.similarity_threshold:
                            best_similarity = similarity
                            best_match_id = track_id
                            best_match_camera = int(cam_id)
        
        if best_match_id:
            return best_match_id, best_similarity, best_match_camera
        return None
    
    def _update_track_camera(self, track_id, features, bbox, camera_id, frame_id):
        """기존 트랙에 새로운 카메라 정보 추가/업데이트 (통합된 구조)"""
        data_key = self.track_data_key_pattern.format(track_id)
        track_data = self.redis_client.get(data_key)
        
        if track_data:
            track_info = pickle.loads(track_data)
        else:
            track_info = {'cameras': {}, 'is_disappeared': False, 'disappeared_since': None, 'last_activity': frame_id}
        
        camera_id_str = str(camera_id)
        
        # 카메라 정보 초기화 또는 업데이트
        if camera_id_str not in track_info['cameras']:
            track_info['cameras'][camera_id_str] = {
                'features': [],
                'last_seen': frame_id,
                'last_bbox': bbox
            }
        
        # 특징 추가 (사라진 객체가 다시 나타난 경우 특징 복구)
        if features is not None:
            track_info['cameras'][camera_id_str]['features'].append(features)
            
            # 슬라이딩 윈도우 적용
            if len(track_info['cameras'][camera_id_str]['features']) > self.max_features_per_camera:
                track_info['cameras'][camera_id_str]['features'] = track_info['cameras'][camera_id_str]['features'][-self.max_features_per_camera:]
        
        # 메타데이터 업데이트
        track_info['cameras'][camera_id_str]['last_seen'] = frame_id
        track_info['cameras'][camera_id_str]['last_bbox'] = bbox
        track_info['last_activity'] = frame_id
        
        # 사라진 객체가 다시 나타난 경우 상태 복구
        if track_info.get('is_disappeared', False) and features is not None:
            track_info['is_disappeared'] = False
            track_info['disappeared_since'] = None
            print(f"Global ReID: Restored disappeared track {track_id} to normal state")
        
        # TTL 결정: 활성 객체는 TTL 없음, 사라진 객체만 TTL
        if track_info.get('is_disappeared', False):
            # 사라진 객체: TTL 적용 (프레임 단위위)
            ttl = self.feature_ttl
            self.redis_client.setex(data_key, ttl, pickle.dumps(track_info))
            track_key = self.track_key_pattern.format(track_id)
            self.redis_client.setex(track_key, ttl, b'1')
        else:
            # 활성 객체: TTL 없음
            self.redis_client.set(data_key, pickle.dumps(track_info))
            track_key = self.track_key_pattern.format(track_id)
            self.redis_client.set(track_key, b'1')
    
    def _create_track(self, track_id, features, bbox, camera_id, frame_id, is_disappeared=False):
        """새로운 트랙 생성 (통합된 구조)"""
        track_key = self.track_key_pattern.format(track_id)
        data_key = self.track_data_key_pattern.format(track_id)
        
        # 트랙 데이터 생성
        track_info = {
            'cameras': {
                str(camera_id): {
                    'features': [features] if not is_disappeared else [],
                    'last_seen': frame_id,
                    'last_bbox': bbox
                }
            },
            'is_disappeared': is_disappeared,
            'disappeared_since': frame_id if is_disappeared else None,
            'last_activity': frame_id
        }
        
        # TTL 결정: 활성 객체는 TTL 없음, 사라진 객체만 TTL
        if is_disappeared:
            # 사라진 객체: TTL 적용 (5분)
            ttl = self.feature_ttl
            self.redis_client.setex(track_key, ttl, b'1')
            self.redis_client.setex(data_key, ttl, pickle.dumps(track_info))
        else:
            # 활성 객체: TTL 없음
            self.redis_client.set(track_key, b'1')
            self.redis_client.set(data_key, pickle.dumps(track_info))
    
    def get_track_info(self, track_id):
        """트랙 정보 조회"""
        data_key = self.track_data_key_pattern.format(track_id)
        track_data = self.redis_client.get(data_key)
        
        if track_data:
            return pickle.loads(track_data)
        return None
    
    def _mark_track_as_disappeared(self, track_id, bbox, camera_id, frame_id):
        """트랙을 사라진 상태로 표시"""
        data_key = self.track_data_key_pattern.format(track_id)
        track_data = self.redis_client.get(data_key)
        
        if track_data:
            track_info = pickle.loads(track_data)
        else:
            track_info = {'cameras': {}, 'is_disappeared': False, 'disappeared_since': None, 'last_activity': frame_id}
        
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
        
        # 사라진 객체: TTL 적용 (5분)
        ttl = self.feature_ttl
        self.redis_client.setex(data_key, ttl, pickle.dumps(track_info))
        track_key = self.track_key_pattern.format(track_id)
        self.redis_client.setex(track_key, ttl, b'1')
    
    def get_all_tracks(self):
        """모든 트랙 정보 조회 (통합된 구조)"""
        tracks = {}
        
        track_keys = self.redis_client.keys("global_track:*")
        for track_key in track_keys:
            track_id = track_key.decode().split(':')[1]
            tracks[track_id] = self.get_track_info(track_id)
        
        return tracks
    
    def get_tracks_by_camera(self, camera_id):
        """특정 카메라의 트랙들 조회 (통합된 구조)"""
        camera_tracks = {}
        
        track_keys = self.redis_client.keys("global_track:*")
        for track_key in track_keys:
            track_id = track_key.decode().split(':')[1]
            track_info = self.get_track_info(track_id)
            
            if track_info and str(camera_id) in track_info['cameras']:
                camera_tracks[track_id] = track_info
        
        return camera_tracks
    
    def _get_existing_id_for_disappearing_object(self, bbox, camera_id, frame_id, matched_tracks):
        """사라지는 객체의 기존 ID를 찾아서 반환 (통합된 구조)"""
        track_keys = self.redis_client.keys("global_track:*")
        best_match_id = None
        min_distance = float('inf')
        
        for track_key in track_keys:
            track_id = track_key.decode().split(':')[1]
            
            if track_id in matched_tracks:
                continue
            
            data_key = self.track_data_key_pattern.format(track_id)
            track_data = self.redis_client.get(data_key)
            
            if track_data:
                track_info = pickle.loads(track_data)
                
                if str(camera_id) in track_info['cameras']:
                    camera_data = track_info['cameras'][str(camera_id)]
                    
                    # 위치 기반으로 가장 가까운 트랙 찾기
                    if 'last_bbox' in camera_data:
                        last_bbox = camera_data['last_bbox']
                        current_center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                        last_center = [(last_bbox[0] + last_bbox[2]) / 2, (last_bbox[1] + last_bbox[3]) / 2]
                        distance = np.sqrt((current_center[0] - last_center[0])**2 + (current_center[1] - last_center[1])**2)
                        
                        # 사라지는 객체는 더 관대한 거리 제한
                        if distance < 300 and distance < min_distance:
                            min_distance = distance
                            best_match_id = track_id
        
        if best_match_id is not None:
            print(f"Global ReID: Found existing ID {best_match_id} for disappearing object (distance: {min_distance:.1f})")
            # 기존 트랙을 사라진 상태로 업데이트
            self._mark_track_as_disappeared(best_match_id, bbox, camera_id, frame_id)
            matched_tracks.add(best_match_id)
            return best_match_id
        
        # 기존 ID를 찾지 못한 경우 새로운 사라진 트랙 생성
        global_id = BaseTrack.next_id()
        print(f"Global ReID: Created new disappeared ID {global_id} for disappearing object (no existing match found)")
        self._create_track(global_id, None, bbox, camera_id, frame_id, is_disappeared=True)
        return global_id
    
    def _update_track_camera_simple(self, track_id, bbox, camera_id, frame_id):
        """특징 없이 위치 정보만 업데이트 (통합된 구조)"""
        data_key = self.track_data_key_pattern.format(track_id)
        track_data = self.redis_client.get(data_key)
        
        if track_data:
            track_info = pickle.loads(track_data)
        else:
            track_info = {'cameras': {}, 'is_disappeared': False, 'disappeared_since': None, 'last_activity': frame_id}
        
        camera_id_str = str(camera_id)
        
        if camera_id_str not in track_info['cameras']:
            track_info['cameras'][camera_id_str] = {
                'features': [],
                'last_seen': frame_id,
                'last_bbox': bbox
            }
        
        # 특징은 추가하지 않고 위치 정보만 업데이트
        track_info['cameras'][camera_id_str]['last_seen'] = frame_id
        track_info['cameras'][camera_id_str]['last_bbox'] = bbox
        track_info['last_activity'] = frame_id
        
        # TTL 결정: 활성 객체는 TTL 없음, 사라진 객체만 TTL
        if track_info.get('is_disappeared', False):
            # 사라진 객체: TTL 적용 (5분)
            ttl = self.feature_ttl
            self.redis_client.setex(data_key, ttl, pickle.dumps(track_info))
            track_key = self.track_key_pattern.format(track_id)
            self.redis_client.setex(track_key, ttl, b'1')
        else:
            # 활성 객체: TTL 없음
            self.redis_client.set(data_key, pickle.dumps(track_info))
            track_key = self.track_key_pattern.format(track_id)
            self.redis_client.set(track_key, b'1')
    
    def _create_track_simple(self, track_id, bbox, camera_id, frame_id):
        """특징 없이 새로운 트랙 생성 (통합된 구조)"""
        return self._create_track(track_id, None, bbox, camera_id, frame_id, is_disappeared=False)
    
    def _create_disappeared_track(self, track_id, bbox, camera_id, frame_id):
        """사라진 객체용 새로운 트랙 생성 (통합된 구조)"""
        return self._create_track(track_id, None, bbox, camera_id, frame_id, is_disappeared=True)
    
    def _update_disappeared_track_camera(self, track_id, bbox, camera_id, frame_id):
        """사라진 객체 트랙 업데이트 (통합된 구조)"""
        return self._update_track_camera_simple(track_id, bbox, camera_id, frame_id) 