import redis
import json
import numpy as np
import pickle
import threading
import time
from scipy.spatial.distance import cdist
from ByteTrack.yolox.tracker.basetrack import BaseTrack

class RedisGlobalReIDManagerV2:
    """
    카메라별 우선순위 매칭과 다중 카메라 정보 통합을 지원하는 Redis Global ReID 매니저
    사라지는 객체 감지 및 TTL 제외 관리 포함
    """
    def __init__(self, similarity_threshold=0.7, feature_ttl=300, redis_host='localhost', redis_port=6379, frame_rate=30, grace_period_frames=15, max_features_per_track=10):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
        self.similarity_threshold = similarity_threshold
        self.feature_ttl = feature_ttl  # 초 단위
        self.frame_rate = frame_rate
        self.grace_period_frames = grace_period_frames  # 새로운 객체의 유예기간 (프레임 단위)
        self.max_features_per_track = max_features_per_track  # 객체당 최대 특징 벡터 수 (유사도 계산 시 사용)
        self.lock = threading.Lock()
        
        # 새로운 Redis 키 패턴 (timestamp 기반)
        self.track_key_pattern = "global_track:{}"  # 트랙 존재 플래그
        self.feature_key_pattern = "track_feature:{}:{}:{}"  # track_id:camera_id:timestamp
        self.track_meta_key_pattern = "track_meta:{}"  # 트랙 메타데이터
        
        # Redis 키 패턴 (사전 등록 데이터용)
        self.pre_registered_key_pattern = "global_track_pre:{}"
        self.pre_registered_data_key_pattern = "global_track_data_pre:{}"
        
        # 전역 프레임 카운터 (모든 카메라가 공유)
        self.global_frame_counter = 0
        
        # TTL을 프레임 단위로 변환
        self.ttl_frames = self.feature_ttl * self.frame_rate
        
        # 카메라별 추적 상태 관리
        self.camera_tracking_ids = {}  # {camera_id: set(track_ids)} - 현재 프레임에서 추적 중인 ID들
        self.camera_lost_ids = {}      # {camera_id: set(track_ids)} - 이전 프레임에서 놓친 ID들
    
    def update_frame(self, frame_id):
        """현재 프레임 업데이트 및 만료된 트랙 정리 (timestamp 기반 구조)"""
        with self.lock:
            # 전역 프레임 카운터 업데이트
            self.global_frame_counter = max(self.global_frame_counter, frame_id)
            
            track_keys = self.redis_client.keys("global_track:*")
            
            for track_key in track_keys:
                track_id = track_key.decode().split(':')[1]
                meta_key = self.track_meta_key_pattern.format(track_id)
                
                # 트랙 메타데이터 가져오기
                meta_data = self.redis_client.get(meta_key)
                if meta_data:
                    track_meta = pickle.loads(meta_data)
                    
                    # 모든 카메라의 마지막 업데이트 시간 확인
                    max_last_seen = 0
                    for camera_data in track_meta['cameras'].values():
                        max_last_seen = max(max_last_seen, camera_data.get('last_seen', 0))
                    
                    # TTL 결정 (시간 기반)
                    is_disappeared = track_meta.get('is_disappeared', False)
                    if is_disappeared:
                        ttl_frames = self.ttl_frames  # 사라진 객체: 기본 TTL
                    else:
                        ttl_frames = self.ttl_frames * 2  # 활성 객체: 2배 TTL
                    
                    # TTL이 만료된 트랙 제거 (시간 기반)
                    if self.global_frame_counter - max_last_seen > ttl_frames:
                        self._remove_track(track_id)
                        status = "disappeared" if is_disappeared else "normal"
                        print(f"Global ReID: Expired {status} track {track_id} (global_frame: {self.global_frame_counter}, last_seen: {max_last_seen}, diff: {self.global_frame_counter - max_last_seen})")
      
    def _remove_track(self, track_id):
        """트랙 완전 제거 (timestamp 기반 구조)"""
        # 트랙 존재 플래그 제거
        track_key = self.track_key_pattern.format(track_id)
        
        # 트랙 메타데이터 제거
        meta_key = self.track_meta_key_pattern.format(track_id)
        
        # 해당 트랙의 모든 특징 벡터 키 찾기
        feature_pattern = self.feature_key_pattern.format(track_id, "*", "*")
        feature_keys = self.redis_client.keys(feature_pattern)
        
        # 모든 관련 키 삭제
        keys_to_delete = [track_key, meta_key] + feature_keys
        if keys_to_delete:
            self.redis_client.delete(*keys_to_delete)
        
        print(f"Global ReID: Removed track {track_id} and {len(feature_keys)} feature vectors")
    
    def detect_disappearing_object(self, bbox, frame_shape, previous_bbox_area=None):
        """객체가 화면에서 사라지는지 감지"""
        x1, y1, x2, y2 = bbox
        frame_height, frame_width = frame_shape #바운딩 박스도 resize된 프레임크기에 맞추어져 있기 때문
        
        # 바운딩 박스가 화면 경계에 닿는지 확인
        touching_boundary = (x1 <= 0 or y1 <= 0 or x2 >= frame_width or y2 >= frame_height)
        
        # 바운딩 박스 크기가 급격히 작아지는지 확인
        bbox_area = (x2 - x1) * (y2 - y1)
        is_shrinking = False
        if previous_bbox_area:
            is_shrinking = bbox_area < previous_bbox_area * 0.5
        
        return touching_boundary or is_shrinking #둘 중 하나라도 True면 사라지는 객체로 판단
    
    def get_disappearing_zone(self, bbox, frame_shape):
        """객체가 사라지는 영역을 구분 (디버깅에 사용중중)"""
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
            # 전역 프레임 카운터 업데이트
            self.global_frame_counter = max(self.global_frame_counter, frame_id)
            
            # 1단계: 사라지는 객체 감지
            is_disappearing = self.detect_disappearing_object(bbox, frame_shape, None)
            
            if is_disappearing:
                print(f"Global ReID: Skipping ReID for disappearing object at {self.get_disappearing_zone(bbox, frame_shape)}")
                # 사라지는 객체는 기존 ID 유지 (ReID 스킵)
                return self._get_existing_id_for_disappearing_object(bbox, camera_id, frame_id, matched_tracks)
            
            # 2단계: 통합 ReID 매칭 (최근 10프레임 + 사전 등록된 원본 데이터)
            # 특징 벡터 정규화
            features = features / np.linalg.norm(features)
            
            best_match_id = None
            best_similarity = 0
            best_match_camera = None
            match_type = None
            
            # 통합 매칭: 최근 10프레임 + 사전 등록된 원본 데이터
            integrated_match = self._match_integrated_features(features, bbox, camera_id, frame_id, matched_tracks)
            if integrated_match:
                best_match_id, best_similarity, best_match_camera, match_type = integrated_match
                print(f"Global ReID: Integrated match - Track {best_match_id} (similarity: {best_similarity:.3f}, type: {match_type})")
            
            if best_match_id is not None:
                # 매칭 타입에 따른 처리
                if match_type in ['pre_registered_priority', 'lost_pre_registered', 'final_pre_registered']:
                    # 사전 등록된 데이터와 매칭된 경우 특별 처리
                    self._activate_pre_registered_track(best_match_id, features, bbox, camera_id, frame_id)
                    print(f"Global ReID: Activated pre-registered track {best_match_id} for camera {camera_id} ({match_type} match)")
                elif match_type == 'origindata':
                    # 사전 등록된 원본 데이터와 매칭된 경우 특별 처리
                    self._activate_pre_registered_track(best_match_id, features, bbox, camera_id, frame_id)
                    print(f"Global ReID: Activated pre-registered track {best_match_id} for camera {camera_id} (origindata match)")
                else:
                    # 기존 트랙에 현재 카메라 정보 추가/업데이트
                    self._update_track_camera(best_match_id, features, bbox, camera_id, frame_id)
                
                matched_tracks.add(best_match_id)
                return best_match_id
            else:
                # 3단계: 유예기간 처리 - 새로운 객체인지 확인
                grace_track_id = self._check_grace_period_track(features, bbox, camera_id, frame_id)
                
                if grace_track_id is not None:
                    # 유예기간 중인 트랙이 있으면 업데이트
                    self._update_grace_period_track(grace_track_id, features, bbox, camera_id, frame_id)
                    print(f"Global ReID: Updated grace period track {grace_track_id} (waiting for {self.grace_period_frames} frames)")
                    return None  # 아직 ID 할당하지 않음
                else:
                    # 새로운 유예기간 트랙 생성
                    grace_track_id = self._create_grace_period_track(features, bbox, camera_id, frame_id)
                    print(f"Global ReID: Created grace period track {grace_track_id} (waiting for {self.grace_period_frames} frames)")
                    return None  # 아직 ID 할당하지 않음
    

    
    def _match_integrated_features(self, features, bbox, camera_id, frame_id, matched_tracks):
        """개선된 통합 매칭: timestamp 기반 구조를 사용한 효율적인 매칭"""
        track_keys = self.redis_client.keys("global_track:*")
        
        best_match_id = None
        best_similarity = 0
        best_match_camera = None
        match_type = None
        
        # 1단계: 같은 카메라의 최근 특징과 매칭 (사전 등록된 ID는 제외)
        for track_key in track_keys:
            track_id = track_key.decode().split(':')[1]
            
            if track_id in matched_tracks:
                continue
            
            # 사전 등록된 ID인지 확인 - 사전 등록된 ID는 1단계에서 제외
            pre_track_info = self.get_pre_registered_track_info(track_id)
            if pre_track_info:
                print(f"Global ReID: Skipping pre-registered ID {track_id} in step 1 (same camera matching)")
                continue
            
            # 트랙 메타데이터 가져오기
            meta_key = self.track_meta_key_pattern.format(track_id)
            meta_data = self.redis_client.get(meta_key)
            
            if meta_data:
                track_meta = pickle.loads(meta_data)
                
                # 같은 카메라의 최근 특징과 매칭
                camera_id_str = str(camera_id)
                if camera_id_str in track_meta['cameras']:
                    camera_data = track_meta['cameras'][camera_id_str]
                    
                    # 위치 기반 필터링 (같은 카메라에서만)
                    location_score = 0
                    if 'last_bbox' in camera_data and camera_data['last_bbox'] is not None:
                        last_bbox = camera_data['last_bbox']
                        current_center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                        last_center = [(last_bbox[0] + last_bbox[2]) / 2, (last_bbox[1] + last_bbox[3]) / 2]
                        distance = np.sqrt((current_center[0] - last_center[0])**2 + (current_center[1] - last_center[1])**2)
                        
                        max_distance = 100
                        if distance <= max_distance:
                            location_score = 1.0 - (distance / max_distance)
                        else:
                            continue
                    else:
                        location_score = 0.5
                    
                    # 최근 특징과 유사도 계산 (새로운 구조 사용)
                    feature_similarity = self._calculate_feature_similarity(features, track_id, camera_id)
                    
                    dynamic_threshold = self.similarity_threshold * (1.0 - location_score * 0.5)
                    dynamic_threshold = max(dynamic_threshold, self.similarity_threshold * 0.3)
                    
                    if feature_similarity > best_similarity and feature_similarity > dynamic_threshold:
                        best_similarity = feature_similarity
                        best_match_id = track_id
                        best_match_camera = camera_id
                        match_type = 'smooth_feat'
                        print(f"Global ReID: Same camera match - Track {track_id} (similarity: {best_similarity:.3f})")
        
        # 2단계: 사전 등록된 데이터와 우선 매칭 (놓친 ID 포함)
        print(f"Global ReID: Step 2 - Checking pre-registered data first...")
        
        # 모든 사전 등록된 데이터 확인
        pre_registered_keys = self.redis_client.keys("global_track_pre:*")
        for pre_key in pre_registered_keys:
            pre_track_id = pre_key.decode().split(':')[1]
            
            if pre_track_id in matched_tracks:
                continue
            
            pre_track_info = self.get_pre_registered_track_info(pre_track_id)
            
            if pre_track_info and 'features' in pre_track_info:
                pre_registered_features = pre_track_info['features']
                
                if len(pre_registered_features) > 0:
                    # 사전 등록된 특징들과 유사도 계산 (단순 평균 사용)
                    features_array = np.array(pre_registered_features)
                    if len(features_array) == 1:
                        average_feature = features_array[0]
                    else:
                        average_feature = np.mean(features_array, axis=0)
                    
                    feature_similarity = 1 - cdist([features], [average_feature], 'cosine')[0][0]
                    pre_registered_threshold = 0.6  # 사전 데이터 우선 매칭
                    
                    if feature_similarity > best_similarity and feature_similarity >= pre_registered_threshold:
                        best_similarity = feature_similarity
                        best_match_id = pre_track_id
                        best_match_camera = 'pre_registered'
                        match_type = 'pre_registered_priority'
                        print(f"Global ReID: Pre-registered priority match - Track {best_match_id} (similarity: {best_similarity:.3f})")
        
        # 3단계: 놓친 ID 중에서 사전 데이터와 매칭 (추가 확인)
        camera_id_str = str(camera_id)
        lost_ids = self.camera_lost_ids.get(camera_id_str, set())
        
        if lost_ids:
            print(f"Global ReID: Checking {len(lost_ids)} lost IDs for pre-registered match")
            for lost_id in lost_ids:
                print(f"Global ReID: Lost ID {lost_id} - checking if pre-registered...")
                pre_track_info = self.get_pre_registered_track_info(lost_id)
                if pre_track_info and 'features' in pre_track_info:
                    print(f"Global ReID: Lost ID {lost_id} IS pre-registered with {len(pre_track_info['features'])} features")
                else:
                    print(f"Global ReID: Lost ID {lost_id} is NOT pre-registered")
            
            for lost_id in lost_ids:
                if lost_id in matched_tracks:
                    continue
                
                # 놓친 ID가 사전 등록된 데이터인지 확인
                pre_track_info = self.get_pre_registered_track_info(lost_id)
                
                if pre_track_info and 'features' in pre_track_info:
                    pre_registered_features = pre_track_info['features']
                    
                    if len(pre_registered_features) > 0:
                        # 사전 등록된 특징들과 유사도 계산 (단순 평균 사용)
                        features_array = np.array(pre_registered_features)
                        if len(features_array) == 1:
                            average_feature = features_array[0]
                        else:
                            average_feature = np.mean(features_array, axis=0)
                        
                        feature_similarity = 1 - cdist([features], [average_feature], 'cosine')[0][0]
                        pre_registered_threshold = 0.6  # 임계값 낮춤
                        
                        if feature_similarity > best_similarity and feature_similarity >= pre_registered_threshold:
                            best_similarity = feature_similarity
                            best_match_id = lost_id
                            best_match_camera = 'pre_registered'
                            match_type = 'lost_pre_registered'
                            print(f"Global ReID: Lost pre-registered match - Track {best_match_id} (similarity: {best_similarity:.3f})")
        
        # 4단계: 다른 카메라의 최근 특징과 매칭
        for track_key in track_keys:
            track_id = track_key.decode().split(':')[1]
            
            if track_id in matched_tracks:
                continue
            
            # 트랙 메타데이터 가져오기
            meta_key = self.track_meta_key_pattern.format(track_id)
            meta_data = self.redis_client.get(meta_key)
            
            if meta_data:
                track_meta = pickle.loads(meta_data)
                
                for cam_id, camera_data in track_meta['cameras'].items():
                    if int(cam_id) == camera_id:  # 같은 카메라는 이미 처리됨
                        continue
                    
                    # 다른 카메라의 최근 특징과 유사도 계산
                    feature_similarity = self._calculate_feature_similarity(features, track_id, int(cam_id))
                    
                    if feature_similarity > best_similarity and feature_similarity > self.similarity_threshold:
                        best_similarity = feature_similarity
                        best_match_id = track_id
                        best_match_camera = int(cam_id)
                        match_type = 'cross_camera'
                        print(f"Global ReID: Cross camera match - Track {track_id} from camera {cam_id} (similarity: {feature_similarity:.3f})")
        
        # 5단계: 사전 데이터 전체와 매칭 (최후 수단)
        if best_match_id is None:
            print(f"Global ReID: No matches found, checking all pre-registered data...")
            pre_registered_keys = self.redis_client.keys("global_track_pre:*")
            print(f"Global ReID: Found {len(pre_registered_keys)} pre-registered keys in Redis")
            for pre_key in pre_registered_keys:
                pre_track_id = pre_key.decode().split(':')[1]
                print(f"Global ReID: Checking pre-registered ID {pre_track_id}")
            
            pre_registered_keys = self.redis_client.keys("global_track_pre:*")
            for pre_key in pre_registered_keys:
                pre_track_id = pre_key.decode().split(':')[1]
                
                if pre_track_id in matched_tracks:
                    continue
                
                pre_track_info = self.get_pre_registered_track_info(pre_track_id)
                
                if pre_track_info and 'features' in pre_track_info:
                    pre_registered_features = pre_track_info['features']
                    
                    if len(pre_registered_features) > 0:
                        # 사전 등록된 특징들과 유사도 계산 (단순 평균 사용)
                        features_array = np.array(pre_registered_features)
                        if len(features_array) == 1:
                            average_feature = features_array[0]
                        else:
                            average_feature = np.mean(features_array, axis=0)
                        
                        feature_similarity = 1 - cdist([features], [average_feature], 'cosine')[0][0]
                        pre_registered_threshold = 0.65  # 최후 수단이므로 더 높은 임계값 (하지만 여전히 낮춤)
                        
                        if feature_similarity > best_similarity and feature_similarity >= pre_registered_threshold:
                            best_similarity = feature_similarity
                            best_match_id = pre_track_id
                            best_match_camera = 'pre_registered'
                            match_type = 'final_pre_registered'
                            print(f"Global ReID: Final pre-registered match - Track {best_match_id} (similarity: {best_similarity:.3f})")
        
        if best_match_id:
            return best_match_id, best_similarity, best_match_camera, match_type
        return None
    
    def _update_track_camera(self, track_id, features, bbox, camera_id, frame_id):
        """기존 트랙에 새로운 카메라 정보 추가/업데이트 (timestamp 기반 구조)"""
        with self.lock:
            # 현재 timestamp 생성
            timestamp = int(time.time() * 1000)  # 밀리초 단위
            
            # 특징 벡터를 개별 키로 저장
            feature_key = self.feature_key_pattern.format(track_id, camera_id, timestamp)
            feature_data = {
                'features': features.tolist() if hasattr(features, 'tolist') else features,
                'bbox': bbox,
                'frame_id': frame_id,
                'timestamp': timestamp
            }
            
            # 특징 벡터 저장 (TTL 적용)
            self.redis_client.setex(feature_key, self.feature_ttl, pickle.dumps(feature_data))
            
            # 트랙 메타데이터 업데이트
            meta_key = self.track_meta_key_pattern.format(track_id)
            meta_data = self.redis_client.get(meta_key)
            
            if meta_data:
                track_meta = pickle.loads(meta_data)
            else:
                track_meta = {
                    'cameras': {},
                    'is_disappeared': False,
                    'disappeared_since': None,
                    'last_activity': frame_id,
                    'created_at': timestamp
                }
            
            camera_id_str = str(camera_id)
            
            # 카메라 정보 업데이트
            if camera_id_str not in track_meta['cameras']:
                track_meta['cameras'][camera_id_str] = {
                    'last_seen': frame_id,
                    'last_bbox': bbox,
                    'feature_count': 0
                }
            
            # 메타데이터 업데이트
            track_meta['cameras'][camera_id_str]['last_seen'] = self.global_frame_counter
            track_meta['cameras'][camera_id_str]['last_bbox'] = bbox
            track_meta['cameras'][camera_id_str]['feature_count'] += 1
            track_meta['last_activity'] = self.global_frame_counter
            
            # 사라진 객체가 다시 나타난 경우 상태 복구
            if track_meta.get('is_disappeared', False):
                track_meta['is_disappeared'] = False
                track_meta['disappeared_since'] = None
                print(f"Global ReID: Restored disappeared track {track_id} to normal state")
            
            # 트랙 메타데이터 저장
            if track_meta.get('is_disappeared', False):
                # 사라진 객체: TTL 적용
                self.redis_client.setex(meta_key, self.feature_ttl, pickle.dumps(track_meta))
                track_key = self.track_key_pattern.format(track_id)
                self.redis_client.setex(track_key, self.feature_ttl, b'1')
            else:
                # 활성 객체: TTL 없음
                self.redis_client.set(meta_key, pickle.dumps(track_meta))
                track_key = self.track_key_pattern.format(track_id)
                self.redis_client.set(track_key, b'1')
    
    def _create_track(self, track_id, features, bbox, camera_id, frame_id, is_disappeared=False):
        """새로운 트랙 생성 (통합된 구조)"""
        track_key = self.track_key_pattern.format(track_id)
        data_key = self.track_data_key_pattern.format(track_id)
        
        # 트랙 데이터 생성 (전역 프레임 카운터 사용)
        track_info = {
            'cameras': {
                str(camera_id): {
                    'features': [features] if not is_disappeared else [],
                    'last_seen': self.global_frame_counter,
                    'last_bbox': bbox
                }
            },
            'is_disappeared': is_disappeared,
            'disappeared_since': self.global_frame_counter if is_disappeared else None,
            'last_activity': self.global_frame_counter
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
    
    def get_pre_registered_track_info(self, track_id):
        """사전 등록된 트랙의 정보 조회 (단순화된 구조)"""
        data_key = self.pre_registered_data_key_pattern.format(track_id)
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
    
    def _get_next_available_id(self):
        """사전 등록된 ID와 충돌하지 않는 다음 ID 생성"""
        pre_registered_keys = self.redis_client.keys("global_track_pre:*")
        track_keys = self.redis_client.keys("global_track:*")
        existing_ids = set()

                # 사전 등록된 ID들도 수집 (충돌 방지)
        for pre_key in pre_registered_keys:
            pre_track_id = pre_key.decode().split(':')[1]
            try:
                existing_ids.add(int(pre_track_id))
            except ValueError:
                # 숫자가 아닌 ID는 무시
                continue

        # 기존 트랙 ID들 수집
        for track_key in track_keys:
            track_id = track_key.decode().split(':')[1]
            try:
                existing_ids.add(int(track_id))
            except ValueError:
                # 숫자가 아닌 ID는 무시
                continue
        
        # 다음 사용 가능한 ID 찾기
        next_id = 1
        while next_id in existing_ids:
            next_id += 1
        
        print(f"Global ReID: Generated new ID {next_id} (existing IDs: {sorted(existing_ids)})")
        return next_id
    
    def _activate_pre_registered_track(self, track_id, features, bbox, camera_id, frame_id):
        """사전 등록된 트랙을 현재 카메라로 활성화"""
        data_key = self.track_data_key_pattern.format(track_id)
        track_data = self.redis_client.get(data_key)
        
        if track_data:
            track_info = pickle.loads(track_data)
        else:
            return
        
        camera_id_str = str(camera_id)
        
        # 현재 카메라 정보 초기화
        if camera_id_str not in track_info['cameras']:
            track_info['cameras'][camera_id_str] = {
                'features': [],
                'last_seen': frame_id,
                'last_bbox': bbox
            }
        
        # 특징 추가
        if features is not None:
            track_info['cameras'][camera_id_str]['features'].append(features)
            
            # 카메라당 특징 벡터 수 제한 없음 - 모든 특징 벡터 유지
        
        # 메타데이터 업데이트
        track_info['cameras'][camera_id_str]['last_seen'] = self.global_frame_counter
        track_info['cameras'][camera_id_str]['last_bbox'] = bbox
        track_info['last_activity'] = self.global_frame_counter
        
        # 사전 등록 플래그는 유지하되 활성 상태로 변경
        track_info['is_pre_registered'] = True  # 사전 등록된 특징은 유지
        track_info['is_disappeared'] = False
        track_info['disappeared_since'] = None
        
        # 활성 객체로 저장 (TTL 없음)
        self.redis_client.set(data_key, pickle.dumps(track_info))
        track_key = self.track_key_pattern.format(track_id)
        self.redis_client.set(track_key, b'1')
    
    # 유예기간 관련 함수들
    def _check_grace_period_track(self, features, bbox, camera_id, frame_id):
        """유예기간 중인 트랙이 있는지 확인하고 유사도가 높은 트랙 ID 반환"""
        grace_keys = self.redis_client.keys("grace_period:*")
        
        best_match_id = None
        best_similarity = 0
        
        for grace_key in grace_keys:
            grace_id = grace_key.decode().split(':')[1]
            data_key = f"grace_period_data:{grace_id}"
            grace_data = self.redis_client.get(data_key)
            
            if grace_data:
                grace_info = pickle.loads(grace_data)
                
                # 같은 카메라의 유예기간 트랙만 확인
                if grace_info.get('camera_id') == camera_id:
                    # 위치 기반 필터링 (같은 카메라에서만)
                    grace_bbox = grace_info.get('last_bbox')
                    if grace_bbox:
                        # IoU 기반 위치 필터링
                        iou = self._calculate_iou(bbox, grace_bbox)
                        if iou > 0.3:  # 위치가 비슷한 경우만
                            # 특징 유사도 계산
                            grace_features = grace_info.get('features', [])
                            if len(grace_features) > 0:
                                # 가장 최근 특징과 비교
                                latest_feature = grace_features[-1]
                                similarity = np.dot(features, latest_feature) / (np.linalg.norm(features) * np.linalg.norm(latest_feature))
                                
                                if similarity > best_similarity and similarity > 0.6:  # 유예기간 중에는 더 관대한 임계값
                                    best_similarity = similarity
                                    best_match_id = grace_id
        
        return best_match_id
    
    def _create_grace_period_track(self, features, bbox, camera_id, frame_id):
        """새로운 유예기간 트랙 생성"""
        grace_id = f"grace_{camera_id}_{frame_id}_{int(time.time())}"
        
        grace_info = {
            'camera_id': camera_id,
            'features': [features],
            'bboxes': [bbox],
            'frame_ids': [frame_id],
            'created_at': frame_id,
            'last_bbox': bbox,
            'last_frame': frame_id,
            'frame_count': 1
        }
        
        # Redis에 저장 (TTL: 30초)
        grace_key = f"grace_period:{grace_id}"
        data_key = f"grace_period_data:{grace_id}"
        
        self.redis_client.setex(grace_key, 30, b'1')
        self.redis_client.setex(data_key, 30, pickle.dumps(grace_info))
        
        return grace_id
    
    def _update_grace_period_track(self, grace_id, features, bbox, camera_id, frame_id):
        """유예기간 트랙 업데이트"""
        data_key = f"grace_period_data:{grace_id}"
        grace_data = self.redis_client.get(data_key)
        
        if grace_data:
            grace_info = pickle.loads(grace_data)
            
            # 특징과 바운딩 박스 추가
            grace_info['features'].append(features)
            grace_info['bboxes'].append(bbox)
            grace_info['frame_ids'].append(frame_id)
            grace_info['last_bbox'] = bbox
            grace_info['last_frame'] = frame_id
            grace_info['frame_count'] += 1
            
            # 유예기간이 지났는지 확인
            if grace_info['frame_count'] >= self.grace_period_frames:
                # 유예기간 완료 - 실제 트랙으로 변환
                self._convert_grace_to_track(grace_id, grace_info)
            else:
                # 유예기간 계속 - 데이터 업데이트
                self.redis_client.setex(data_key, 30, pickle.dumps(grace_info))
    
    def _convert_grace_to_track(self, grace_id, grace_info):
        """유예기간 트랙을 실제 트랙으로 변환"""
        # 새로운 글로벌 ID 생성
        global_id = self._get_next_available_id()
        
        # 가장 최근 특징을 사용하여 트랙 생성
        latest_features = grace_info['features'][-1]
        latest_bbox = grace_info['last_bbox']
        camera_id = grace_info['camera_id']
        frame_id = grace_info['last_frame']
        
        # 실제 트랙 생성
        self._create_track(global_id, latest_features, latest_bbox, camera_id, frame_id)
        
        # 유예기간 데이터 삭제
        grace_key = f"grace_period:{grace_id}"
        data_key = f"grace_period_data:{grace_id}"
        self.redis_client.delete(grace_key, data_key)
        
        print(f"Global ReID: Grace period completed for {grace_id} → Converted to track {global_id}")
    
    def _calculate_iou(self, bbox1, bbox2):
        """두 바운딩 박스의 IoU 계산"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 교집합 영역 계산
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 합집합 영역 계산
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0 

    def _get_recent_features(self, track_id, camera_id=None, limit=None):
        """트랙의 최근 특징 벡터들을 가져옴 (timestamp 기반)"""
        if limit is None:
            limit = self.max_features_per_track
        
        # 특징 벡터 키 패턴으로 검색
        if camera_id is not None:
            pattern = self.feature_key_pattern.format(track_id, camera_id, "*")
        else:
            pattern = self.feature_key_pattern.format(track_id, "*", "*")
        
        feature_keys = self.redis_client.keys(pattern)
        
        if not feature_keys:
            return []
        
        # timestamp로 정렬 (최신순)
        feature_keys.sort(key=lambda x: x.decode().split(':')[-1], reverse=True)
        
        # 최근 limit개만 가져오기
        recent_features = []
        for key in feature_keys[:limit]:
            feature_data = self.redis_client.get(key)
            if feature_data:
                feature_info = pickle.loads(feature_data)
                recent_features.append(feature_info)
        
        return recent_features
    
    def _calculate_feature_similarity(self, current_features, track_id, camera_id=None):
        """현재 특징 벡터와 트랙의 최근 특징 벡터들 간의 유사도 계산"""
        recent_features = self._get_recent_features(track_id, camera_id)
        
        if not recent_features:
            return 0.0
        
        # 최근 특징 벡터들의 평균 계산 (가중 평균 적용)
        feature_vectors = []
        weights = []
        
        for i, feature_info in enumerate(recent_features):
            feature_vector = np.array(feature_info['features'])
            # 최신 특징에 더 높은 가중치 (0.5 ~ 1.0)
            weight = 0.5 + (0.5 * i / len(recent_features))
            feature_vectors.append(feature_vector)
            weights.append(weight)
        
        # 가중 평균 계산
        weights = np.array(weights) / np.sum(weights)
        weighted_average = np.average(feature_vectors, axis=0, weights=weights)
        
        # 코사인 유사도 계산
        current_features_norm = current_features / np.linalg.norm(current_features)
        weighted_average_norm = weighted_average / np.linalg.norm(weighted_average)
        
        similarity = 1 - cdist([current_features_norm], [weighted_average_norm], 'cosine')[0][0]
        return similarity 

 