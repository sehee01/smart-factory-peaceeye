from typing import List, Dict, Optional, Set
from .redis_handler import FeatureStoreRedisHandler
from .similarity import FeatureSimilarityCalculator
from .pre_registration import PreRegistrationManager
from .matchers import PreRegistrationMatcher, SameCameraMatcher, CrossCameraMatcher
from config import settings
import numpy as np
import redis


class GlobalReIDManager:
    """
    객체의 appearance feature를 기반으로 Redis에 저장된 트랙들과 유사도를 계산하여
    글로벌 ID를 재부여하는 책임 클래스.
    원본의 복잡한 로직을 포함: 사라지는 객체 감지, 카메라별 우선순위 매칭, TTL 관리
    """

    def __init__(self,
                 redis_handler: FeatureStoreRedisHandler,
                 similarity_calc: FeatureSimilarityCalculator,
                 similarity_threshold: float = None,
                 feature_ttl: int = None,
                 frame_rate: int = None):
        self.redis = redis_handler
        self.similarity = similarity_calc
        
        # settings.py에서 기본값 가져오기
        if similarity_threshold is None:
            similarity_threshold = settings.REID_CONFIG["threshold"]
        if feature_ttl is None:
            feature_ttl = settings.REID_CONFIG["feature_ttl"]
        if frame_rate is None:
            frame_rate = settings.REID_CONFIG["frame_rate"]
            
        self.threshold = similarity_threshold
        self.feature_ttl = feature_ttl
        self.frame_rate = frame_rate
        self.global_frame_counter = 0
        
        # 사전 등록 매니저 초기화
        self.pre_reg_manager = PreRegistrationManager()
        
        # 매칭 클래스들 초기화
        self.pre_reg_matcher = PreRegistrationMatcher(self.pre_reg_manager, similarity_calc)
        self.same_camera_matcher = SameCameraMatcher(redis_handler, similarity_calc, similarity_threshold)
        self.cross_camera_matcher = CrossCameraMatcher(redis_handler, similarity_calc, similarity_threshold)

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

        best_match_id_same_camera = None
        best_match_id_other_camera = None
        best_similarity_same_camera = 0
        best_similarity_other_camera = 0

        if features is None or len(features) == 0:
            return None
        
        # 1. 사전 등록 기반 매칭 시도 (최우선)
        pre_reg_match = self.pre_reg_matcher.match(features)
        if pre_reg_match:
            print(f"Global ReID: Pre-registration match - Track {pre_reg_match}")
            # 사전 등록된 Global ID로 새로운 track 생성
            self.redis.create_pre_registered_track(pre_reg_match, camera_id, frame_id, 
                                                 features, bbox, local_track_id)
            matched_tracks.add(pre_reg_match)
            return pre_reg_match
        
        # 2. 같은 카메라 내 매칭
        same_camera_match = self.same_camera_matcher.match(features, bbox, camera_id, frame_id, matched_tracks)
        if same_camera_match:
            best_match_id_same_camera, best_similarity_same_camera = same_camera_match
            print(f"Global ReID: Same camera match - Track {best_match_id_same_camera} (similarity: {best_similarity_same_camera:.3f})")
            self._update_track_camera(best_match_id_same_camera, features, bbox, camera_id, frame_id, local_track_id)
            matched_tracks.add(best_match_id_same_camera)
        
        # 3. 다른 카메라 매칭 (낮은 우선순위)
        other_camera_match = self.cross_camera_matcher.match(features, bbox, camera_id, frame_id, matched_tracks)
        if other_camera_match:
            best_match_id_other_camera, best_similarity_other_camera = other_camera_match
            print(f"Global ReID: Cross camera match - Track {best_match_id_other_camera} (similarity: {best_similarity_other_camera:.3f})")
            self._update_track_camera(best_match_id_other_camera, features, bbox, camera_id, frame_id, local_track_id)
            matched_tracks.add(best_match_id_other_camera)
        
        # 4. 최종 매칭 결과 결정
        if best_similarity_same_camera > best_similarity_other_camera:
            return best_match_id_same_camera
        elif best_similarity_other_camera > best_similarity_same_camera:
            return best_match_id_other_camera
        else:
            # 새로운 글로벌 ID 생성
            global_id = self.redis.generate_new_global_id()
            self._create_track(global_id, features, bbox, camera_id, frame_id, local_track_id)
            print(f"Global ReID: Created new track {global_id}")
            return global_id

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
