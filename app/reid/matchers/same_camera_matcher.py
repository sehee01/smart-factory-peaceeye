import numpy as np
from typing import Optional, Set, Dict, Any, List
from ..redis_handler import FeatureStoreRedisHandler
from ..similarity import FeatureSimilarityCalculator
import sys
import os
import logging

# app 디렉토리 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(os.path.dirname(current_dir))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from config import settings

# 로깅 설정
log_level = getattr(logging, settings.LOGGING_CONFIG["level"].upper(), logging.INFO)
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SameCameraMatcher:
    """
    같은 카메라 내에서 객체 매칭을 담당하는 클래스
    """
    
    def __init__(self, redis_handler: FeatureStoreRedisHandler, 
                 similarity_calc: FeatureSimilarityCalculator,
                 similarity_threshold: float = None):
        self.redis = redis_handler
        self.similarity = similarity_calc
        
        # settings.py에서 설정값 가져오기
        if similarity_threshold is None:
            self.threshold = settings.REID_CONFIG["threshold"]
        else:
            self.threshold = similarity_threshold
            
        # same_camera 관련 설정값들
        self.location_threshold = settings.REID_CONFIG["same_camera"]["location_threshold"]
        self.max_distance = settings.REID_CONFIG["same_camera"]["max_distance"]
        self.dynamic_threshold_factor = settings.REID_CONFIG["same_camera"]["dynamic_threshold_factor"]
        self.min_threshold_factor = settings.REID_CONFIG["same_camera"]["min_threshold_factor"]
        self.weight_start = settings.REID_CONFIG["same_camera"]["weight_start"]
        self.weight_end = settings.REID_CONFIG["same_camera"]["weight_end"]
        
        logger.info(f"🔧 SameCameraMatcher 초기화 완료")
        logger.info(f"  - 기본 임계값: {self.threshold}")
        logger.info(f"  - 위치 임계값: {self.location_threshold}")
        logger.info(f"  - 최대 거리: {self.max_distance}px")
        logger.info(f"  - 동적 임계값 계수: {self.dynamic_threshold_factor}")
        logger.info(f"  - 최소 임계값 계수: {self.min_threshold_factor}")
        logger.info(f"  - 가중치 범위: {self.weight_start} ~ {self.weight_end}")
    
    def match(self, features: np.ndarray, bbox: List[int], camera_id: str, 
              frame_id: int, matched_tracks: Set[int]) -> Optional[tuple]:
        """
        같은 카메라 내에서 매칭 수행
        
        Args:
            features: 현재 객체의 feature 벡터
            bbox: 바운딩 박스 [x1, y1, x2, y2]
            camera_id: 카메라 ID
            frame_id: 현재 프레임 ID
            matched_tracks: 이미 매칭된 트랙 ID들의 집합
            
        Returns:
            (global_id, similarity) 튜플 또는 None
        """
        logger.info(f"🎯 SameCameraMatcher 매칭 시작 - Camera: {camera_id}, Frame: {frame_id}")
        logger.info(f"📊 입력 features shape: {features.shape}, bbox: {bbox}")
        logger.info(f"🚫 이미 매칭된 tracks: {matched_tracks}")
        
        candidates = self.redis.get_candidate_features_by_camera(camera_id)
        
        logger.info(f"🔍 Camera {camera_id}에서 {len(candidates)}개 후보 발견")
        
        if not candidates:
            logger.warning("❌ 후보가 없습니다.")
            return None
        
        best_match_id = None
        best_similarity = 0
        
        for global_id, candidate_data in candidates.items():
            logger.info(f"🔍 후보 {global_id} 검사 시작")
            
            if global_id in matched_tracks:
                logger.info(f"⏭️ 후보 {global_id}: 이미 매칭됨 - 건너뜀")
                continue
            
            candidate_features = candidate_data['features']
            candidate_bbox = candidate_data.get('bbox', bbox)
            
            logger.info(f"📊 후보 {global_id}: features 개수={len(candidate_features)}, bbox={candidate_bbox}")
            
            # 위치 기반 필터링 (같은 카메라에서만)
            location_score = self._calculate_location_score(bbox, candidate_bbox)
            logger.info(f"📍 후보 {global_id}: 위치 점수 = {location_score:.4f}")
            
            if location_score < self.location_threshold:
                logger.warning(f"❌ 후보 {global_id}: 위치 점수 {location_score:.4f} < 임계값 {self.location_threshold:.4f} - 필터링됨")
                continue
            
            # 특징 유사도 계산
            if len(candidate_features) > 0:
                logger.info(f"🔍 후보 {global_id}: 특징 유사도 계산 시작")
                
                # 가중 평균 특징 계산
                features_array = np.array(candidate_features)
                if len(features_array) == 1:
                    weighted_average = features_array[0]
                    logger.info(f"📊 후보 {global_id}: 단일 특징 사용")
                else:
                    weights = np.linspace(self.weight_start, self.weight_end, len(features_array))
                    weights = weights / np.sum(weights)
                    weighted_average = np.average(features_array, axis=0, weights=weights)
                    logger.info(f"📊 후보 {global_id}: 가중 평균 특징 계산 (가중치: {weights})")
                
                # 유사도 계산 (컨텍스트 정보 포함)
                context = f"same_camera_{camera_id}_track_{global_id}"
                feature_similarity = self.similarity.calculate_similarity(features, weighted_average, context)
                
                # 동적 임계값 조정 (위치 점수에 따라)
                dynamic_threshold = self.threshold * (1.0 - location_score * self.dynamic_threshold_factor)
                min_threshold = self.threshold * self.min_threshold_factor
                adjusted_threshold = max(dynamic_threshold, min_threshold)
                
                logger.info(f"🎯 후보 {global_id}: 원본 유사도 = {feature_similarity:.4f}")
                logger.info(f"🎯 후보 {global_id}: 기본 임계값 = {self.threshold:.4f}, 동적 임계값 = {adjusted_threshold:.4f}")
                
                # 위치가 가까우면 유사도에 보너스 추가 (위치 점수가 높을수록)
                if location_score > 0.8:  # 위치가 매우 가까우면
                    location_bonus = min(0.1, location_score * 0.1)  # 최대 0.1까지
                    adjusted_similarity = feature_similarity + location_bonus
                    logger.info(f"🎁 후보 {global_id}: 위치 보너스 +{location_bonus:.4f} 적용")
                    logger.info(f"📊 후보 {global_id}: 원본={feature_similarity:.4f}, 보너스=+{location_bonus:.4f}, 조정됨={adjusted_similarity:.4f}, 위치점수={location_score:.4f}")
                else:
                    adjusted_similarity = feature_similarity
                    logger.info(f"📊 후보 {global_id}: 유사도={feature_similarity:.4f}, 동적임계값={adjusted_threshold:.4f}, 위치점수={location_score:.4f}")
                
                if adjusted_similarity > best_similarity and feature_similarity > adjusted_threshold:
                    best_similarity = adjusted_similarity
                    best_match_id = global_id
                    logger.info(f"🏆 후보 {global_id}: 새로운 최고 매치! (조정된 유사도: {adjusted_similarity:.4f})")
                else:
                    if feature_similarity <= adjusted_threshold:
                        logger.warning(f"❌ 후보 {global_id}: 유사도 {feature_similarity:.4f} <= 동적임계값 {adjusted_threshold:.4f}")
                    if adjusted_similarity <= best_similarity:
                        logger.info(f"📉 후보 {global_id}: 조정된 유사도 {adjusted_similarity:.4f} <= 현재 최고 {best_similarity:.4f}")
            else:
                logger.warning(f"❌ 후보 {global_id}: 특징이 비어있음")
        
        if best_match_id:
            logger.info(f"✅ Same camera 매칭 성공: Track {best_match_id} (유사도: {best_similarity:.4f})")
            return best_match_id, best_similarity
        
        logger.warning("❌ Same camera 매칭 실패: 적합한 후보 없음")
        return None
    
    def _calculate_location_score(self, bbox1: List[int], bbox2: List[int]) -> float:
        """두 바운딩 박스 간의 위치 유사도 계산"""
        center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
        center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]
        
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # 거리 기반 위치 점수 계산 (가까울수록 높은 점수)
        if distance <= self.max_distance:
            score = 1.0 - (distance / self.max_distance)  # 0~1 사이 점수
            logger.debug(f"📍 위치 점수 계산: 거리={distance:.2f}, 최대거리={self.max_distance}, 점수={score:.4f}")
            return score
        else:
            logger.debug(f"📍 위치 점수 계산: 거리={distance:.2f} > 최대거리={self.max_distance}, 점수=0.0")
            return 0.0
