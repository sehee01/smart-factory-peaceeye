class LocalReIDManager:
    """
    여러 카메라에서 동일한 객체를 식별하고 ByteTrack ID를 로컬 ID로 관리하는 클래스
    """
    def __init__(self, similarity_threshold=0.6, feature_ttl=300, max_features=10):
        self.local_tracks = {}  # {local_id: {'features': [], 'last_seen': frame_id, 'camera_id': int}}
        self.similarity_threshold = similarity_threshold
        self.feature_ttl = feature_ttl  # 프레임 단위 TTL
        self.max_features = max_features  # 슬라이딩 윈도우 크기
        self.current_frame = 0
    
    def update_frame(self, frame_id):
        """현재 프레임 업데이트 및 만료된 트랙 정리"""
        self.current_frame = frame_id
        
        # TTL이 만료된 트랙 제거
        expired_tracks = []
        for local_id, track_info in self.local_tracks.items():
            if self.current_frame - track_info['last_seen'] > self.feature_ttl:
                expired_tracks.append(local_id)
        
        for local_id in expired_tracks:
            del self.local_tracks[local_id]
            print(f"Local ReID: Expired track {local_id}")
    
    def match_or_create(self, features, bbox, camera_id, frame_id):
        """
        ReID 특징을 기반으로 기존 트랙과 매칭하거나 새로운 로컬 ID 생성
        
        Args:
            features: ReID 특징 벡터 (numpy array)
            bbox: 바운딩 박스 [x1, y1, x2, y2]
            camera_id: 카메라 ID
            frame_id: 현재 프레임 ID
            
        Returns:
            local_id: 매칭된 또는 새로 생성된 로컬 ID
        """
        if features is None or len(features) == 0: #특징 벡터 유무무
            return None
        
        # 특징 벡터 정규화
        features = features / np.linalg.norm(features)
        
        best_match_id = None # 매칭된 로컬 ID
        best_similarity = 0 # 유사도
        
        # 기존 트랙들과 유사도 계산
        for local_id, track_info in self.local_tracks.items():
            if len(track_info['features']) == 0:
                continue
            
            # 가중 평균 특징 계산 (최신 특징에 더 높은 가중치)
            features_array = np.array(track_info['features'])
            if len(features_array) == 1:
                # 특징이 하나뿐인 경우
                weighted_average = features_array[0]
            else:
                # 가중치 계산: 최신 특징에 더 높은 가중치 (0.5 ~ 1.0)
                # 예: 3개 특징이 있으면 [0.5, 0.75, 1.0] -> 정규화 후 [0.22, 0.33, 0.45]
                weights = np.linspace(0.5, 1.0, len(features_array))
                weights = weights / np.sum(weights)  # 정규화
                weighted_average = np.average(features_array, axis=0, weights=weights)
            
            similarity = 1 - cdist([features], [weighted_average], 'cosine')[0][0]
            
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_match_id = local_id
        
        if best_match_id is not None:
            # 기존 트랙과 매칭
            self.local_tracks[best_match_id]['features'].append(features)
            self.local_tracks[best_match_id]['last_seen'] = frame_id
            self.local_tracks[best_match_id]['camera_id'] = camera_id
            
            # 슬라이딩 윈도우 방식으로 특징 개수 제한
            if len(self.local_tracks[best_match_id]['features']) > self.max_features:
                # 가장 오래된 특징부터 제거 (FIFO 방식)
                self.local_tracks[best_match_id]['features'] = self.local_tracks[best_match_id]['features'][-self.max_features:]
            
            print(f"Local ReID: Matched to existing track {best_match_id} (similarity: {best_similarity:.3f}, features: {len(self.local_tracks[best_match_id]['features'])})")
            return best_match_id
        else:
            # 새로운 로컬 ID 생성 (ByteTrack의 next_id() 사용)
            from ByteTrack.yolox.tracker.basetrack import BaseTrack
            local_id = BaseTrack.next_id()
            
            self.local_tracks[local_id] = {
                'features': [features],
                'last_seen': frame_id,
                'camera_id': camera_id
            }
            
            print(f"Local ReID: Created new track {local_id} (features: 1)")
            return local_id