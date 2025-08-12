from typing import List, Dict, Optional, Set
from .redis_handler import FeatureStoreRedisHandler
from .similarity import FeatureSimilarityCalculator
import numpy as np


class GlobalReIDManager:
    """
    객체의 appearance feature를 기반으로 Redis에 저장된 트랙들과 유사도를 계산하여
    글로벌 ID를 재부여하는 책임 클래스.
    원본의 복잡한 로직을 포함: 사라지는 객체 감지, 카메라별 우선순위 매칭, TTL 관리
    """

    def __init__(self,
                 redis_handler: FeatureStoreRedisHandler,
                 similarity_calc: FeatureSimilarityCalculator,
                 similarity_threshold: float = 0.7,
                 feature_ttl: int = 300,
                 frame_rate: int = 30):
        self.redis = redis_handler
        self.similarity = similarity_calc
        self.threshold = similarity_threshold
        self.feature_ttl = feature_ttl
        self.frame_rate = frame_rate
        self.global_frame_counter = 0

    def update_frame(self, frame_id: int):
        """현재 프레임 업데이트 및 만료된 트랙 정리"""
        self.global_frame_counter = max(self.global_frame_counter, frame_id)
        self.redis.cleanup_expired_tracks(self.global_frame_counter, self.feature_ttl)

    def match_or_create(self, features: np.ndarray, bbox: List[int], camera_id: str,
                        frame_id: int, frame_shape: tuple, matched_tracks: Optional[Set[int]] = None,
                        local_track_id: Optional[int] = None) -> Optional[int]:
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
        
        same_camera_match = self._match_same_camera(features, bbox, camera_id, frame_id, matched_tracks)
        if same_camera_match:
            best_match_id, best_similarity = same_camera_match
            print(f"Global ReID: Same camera match - Track {best_match_id} (similarity: {best_similarity:.3f})")
            self._update_track_camera(best_match_id, features, bbox, camera_id, frame_id, local_track_id)
            matched_tracks.add(best_match_id)
            return best_match_id
        
        # 다른 카메라 매칭 (낮은 우선순위)
        other_camera_match = self._match_other_cameras(features, bbox, camera_id, frame_id, matched_tracks)
        if other_camera_match:
            best_match_id, best_similarity = other_camera_match
            print(f"Global ReID: Cross camera match - Track {best_match_id} (similarity: {best_similarity:.3f})")
            self._update_track_camera(best_match_id, features, bbox, camera_id, frame_id, local_track_id)
            matched_tracks.add(best_match_id)
            return best_match_id
        
        # 새로운 글로벌 ID 생성
        global_id = self.redis.generate_new_global_id()
        self._create_track(global_id, features, bbox, camera_id, frame_id, local_track_id)
        print(f"Global ReID: Created new track {global_id}")
        return global_id

    def _get_previous_bbox_area(self, camera_id: str, frame_id: int) -> Optional[float]:
        """이전 프레임의 바운딩 박스 크기 가져오기"""
        # 간단한 구현: 실제로는 더 정교한 히스토리 관리 필요
        return None

    def _match_same_camera(self, features: np.ndarray, bbox: List[int], camera_id: str, 
                          frame_id: int, matched_tracks: Set[int]) -> Optional[tuple]:
        """같은 카메라 내 매칭"""
        candidates = self.redis.get_candidate_features_by_camera(camera_id)
        
        print(f"[DEBUG] Same camera matching for camera {camera_id}: found {len(candidates)} candidates")
        
        best_match_id = None
        best_similarity = 0
        
        for global_id, candidate_data in candidates.items():
            if global_id in matched_tracks:
                continue
            
            candidate_features = candidate_data['features']
            candidate_bbox = candidate_data.get('bbox', bbox)
            
            # 위치 기반 필터링 (같은 카메라에서만) - 더 관대하게
            location_score = self._calculate_location_score(bbox, candidate_bbox)
            if location_score < 0.05:  # 더 관대하게 (0.1 -> 0.05)
                continue #location_score가 현재 0또는 1이여서 BBOX거리100픽셀 이상이면 매칭 안됨
            
            # 특징 유사도 계산
            if len(candidate_features) > 0:
                # 가중 평균 특징 계산
                features_array = np.array(candidate_features)
                if len(features_array) == 1:
                    weighted_average = features_array[0]
                else:
                    weights = np.linspace(0.5, 1.0, len(features_array))
                    weights = weights / np.sum(weights)
                    weighted_average = np.average(features_array, axis=0, weights=weights)
                
                feature_similarity = self.similarity.calculate_similarity(features, weighted_average)
                
                # 위치 기반 동적 임계값 계산 (더 관대하게)
                # 위치가 가까우면 임계값을 낮춤 (더 관대한 매칭)
                dynamic_threshold = self.threshold * (1.0 - location_score * 0.7)
                # 최소 임계값 보장 (더 낮게)
                dynamic_threshold = max(dynamic_threshold, self.threshold * 0.2)
                
                print(f"[DEBUG] Track {global_id}: similarity={feature_similarity:.3f}, threshold={dynamic_threshold:.3f}, location_score={location_score:.3f}")
                
                if feature_similarity > best_similarity and feature_similarity > dynamic_threshold:
                    best_similarity = feature_similarity
                    best_match_id = global_id
                    print(f"[DEBUG] New best match: Track {global_id} (similarity: {feature_similarity:.3f})")
        
        if best_match_id:
            print(f"[DEBUG]  same camera match: Track {best_match_id} (similarity: {best_similarity:.3f})")
            return best_match_id, best_similarity
        return None

    def _match_other_cameras(self, features: np.ndarray, bbox: List[int], camera_id: str, 
                            frame_id: int, matched_tracks: Set[int]) -> Optional[tuple]:
        """다른 카메라와 매칭"""
        candidates = self.redis.get_candidate_features(exclude_camera=camera_id)
        
        best_match_id = None
        best_similarity = 0
        
        for global_id, candidate_features in candidates.items():
            if global_id in matched_tracks:
                continue
            
            if len(candidate_features) > 0:
                # 가중 평균 특징 계산
                features_array = np.array(candidate_features)
                if len(features_array) == 1:
                    weighted_average = features_array[0]
                else:
                    weights = np.linspace(0.5, 1.0, len(features_array))
                    weights = weights / np.sum(weights)
                    weighted_average = np.average(features_array, axis=0, weights=weights)
                
                similarity = self.similarity.calculate_similarity(features, weighted_average)
                
                # 다른 카메라는 더 엄격한 임계값 사용 (더 관대하게)
                if similarity > best_similarity and similarity > self.threshold * 1.0:
                    best_similarity = similarity
                    best_match_id = global_id
        
        if best_match_id:
            print(f"[DEBUG]  other camera match: Track {best_match_id} (similarity: {best_similarity:.3f})")
            return best_match_id, best_similarity
        return None

    def _calculate_location_score(self, bbox1: List[int], bbox2: List[int]) -> float:
        """두 바운딩 박스 간의 위치 유사도 계산"""
        center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
        center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]
        
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # 거리 기반 위치 점수 계산 (가까울수록 높은 점수)
        max_distance = 100
        if distance <= max_distance:
            return 1.0 - (distance / max_distance)  # 0~1 사이 점수
        else:
            return 0.0

    def _update_track_camera(self, global_id: int, features: np.ndarray, bbox: List[int],
                             camera_id: str, frame_id: int, local_track_id: Optional[int] = None):
        """기존 트랙에 새로운 카메라 정보 추가/업데이트"""
        self.redis.store_feature_with_metadata(
            global_id, camera_id, frame_id, features, bbox,
            self.global_frame_counter, local_track_id
        )

    def _create_track(self, global_id: int, features: np.ndarray, bbox: List[int],
                      camera_id: str, frame_id: int, local_track_id: Optional[int] = None):
        """새로운 트랙 생성"""
        self.redis.create_new_track(
            global_id, camera_id, frame_id, features, bbox,
            self.global_frame_counter, local_track_id
        )

    def reassign_global_id(self, camera_id: str, local_track_id: int, feature: np.ndarray) -> int:
        """
        기존 메서드 (하위 호환성을 위해 유지)
        """
        # 간단한 매칭 로직
        candidates = self.redis.get_candidate_features(exclude_camera=camera_id)
        matched_id = self.similarity.find_best_match(feature, candidates, self.threshold)
        
        if matched_id:
            global_id = matched_id
        else:
            global_id = self.redis.generate_new_global_id()
        
        self.redis.store_feature(global_id, camera_id, local_track_id, feature)
        return global_id
