import numpy as np
from typing import Optional, Set, List
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


class CrossCameraMatcher:
    """
    다른 카메라와의 객체 매칭을 담당하는 클래스
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
            
        # cross_camera 관련 설정값들
        self.threshold_cross = settings.REID_CONFIG["cross_camera"]["threshold_cross"]
        self.weight_start = settings.REID_CONFIG["cross_camera"]["weight_start"]
        self.weight_end = settings.REID_CONFIG["cross_camera"]["weight_end"]
        
        
        logger.info(f"🔧 CrossCameraMatcher 초기화 완료")

        logger.info(f"  -  임계값: {self.threshold_cross:.4f}")
        logger.info(f"  - 가중치 범위: {self.weight_start} ~ {self.weight_end}")
    
    def match(self, features: np.ndarray, bbox: List[int], camera_id: str, 
              frame_id: int, matched_tracks: Set[int]) -> Optional[tuple]:
        """
        다른 카메라와 매칭 수행
        
        Args:
            features: 현재 객체의 feature 벡터
            bbox: 바운딩 박스 [x1, y1, x2, y2]
            camera_id: 카메라 ID
            frame_id: 현재 프레임 ID
            matched_tracks: 이미 매칭된 트랙 ID들의 집합
            
        Returns:
            (global_id, similarity) 튜플 또는 None
        """
        logger.info(f"🎯 CrossCameraMatcher 매칭 시작 - Camera: {camera_id}, Frame: {frame_id}")
        logger.info(f"📊 입력 features shape: {features.shape}, bbox: {bbox}")
        logger.info(f"🚫 이미 매칭된 tracks: {matched_tracks}")
        
        candidates = self.redis.get_candidate_features(exclude_camera=camera_id)
        
        logger.info(f"🔍 Camera {camera_id} 제외한 {len(candidates)}개 후보 발견")
        
        if not candidates:
            logger.warning("❌ 다른 카메라에서 후보가 없습니다.")
            return None
        
        best_match_id = None
        best_similarity = 0
        
        for global_id, candidate_data in candidates.items():
            
            if global_id in matched_tracks:
                logger.info(f"⏭️ 후보 {global_id}: 이미 매칭됨 - 건너뜀")
                continue
            
            # candidate_data 구조 확인 및 처리
            if isinstance(candidate_data, dict) and 'features' in candidate_data:
                candidate_features = candidate_data['features']
                candidate_camera = candidate_data.get('camera_id', 'unknown')
                candidate_bbox = candidate_data.get('bbox', bbox)
            else:
                # 명확한 에러 발생 - 데이터 구조가 예상과 다름
                error_msg = f"❌ 예상과 다른 데이터 구조: global_id={global_id}, type={type(candidate_data)}, data={candidate_data}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info(f"📊 후보 {global_id}: features 개수={len(candidate_features)}, camera={candidate_camera}, bbox={candidate_bbox}")
            
            if len(candidate_features) > 0:
                
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
                context = f"cross_camera_{camera_id}_to_{candidate_camera}_track_{global_id}"
                similarity = self.similarity.calculate_similarity(features, weighted_average, context)
                
                logger.info(f"🎯 후보 {global_id}: 유사도 = {similarity:.4f}, cross_camera 임계값 = {self.threshold_cross:.4f}")
                
                # 다른 카메라는 조정된 임계값 사용
                if similarity > best_similarity and similarity > self.threshold_cross:
                    best_similarity = similarity
                    best_match_id = global_id
                    logger.info(f"🏆 후보 {global_id}: 새로운 최고 매치! (유사도: {similarity:.4f})")
                else:
                    if similarity <= self.threshold_cross:
                        logger.warning(f"❌ 후보 {global_id}: 유사도 {similarity:.4f} <= cross_camera 임계값 {self.threshold_cross:.4f}")
                    if similarity <= best_similarity:
                        logger.info(f"📉 후보 {global_id}: 유사도 {similarity:.4f} <= 현재 최고 {best_similarity:.4f}")
            else:
                logger.warning(f"❌ 후보 {global_id}: 특징이 비어있음")
        
        if best_match_id:
            logger.info(f"✅ Cross camera 매칭 성공: Track {best_match_id} (유사도: {best_similarity:.4f})")
            return best_match_id, best_similarity
        else:
            logger.warning(f"❌ Cross camera 매칭 실패: 적합한 후보 없음 (최고 유사도: {best_similarity:.4f})")
        return None
