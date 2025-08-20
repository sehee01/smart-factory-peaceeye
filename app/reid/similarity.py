from scipy.spatial.distance import cdist
import numpy as np
from typing import Dict, Optional
import logging
import sys
import os
from app.config import settings

# app 디렉토리 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(os.path.dirname(current_dir))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# 로깅 설정
log_level = getattr(logging, settings.LOGGING_CONFIG["level"].upper(), logging.INFO)
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureSimilarityCalculator:
    """
    feature 벡터 간의 cosine 유사도를 기반으로 가장 유사한 글로벌 ID를 계산하는 유틸리티
    """

    def __init__(self, enable_detailed_logging: bool = None):
        if enable_detailed_logging is None:
            self.enable_detailed_logging = settings.LOGGING_CONFIG.get("similarity_detailed_logging", True)
        else:
            self.enable_detailed_logging = enable_detailed_logging
        self.similarity_history = []  # 유사도 계산 히스토리 저장

    def find_best_match(self,
                        feature: np.ndarray,
                        candidates: Dict[int, np.ndarray],
                        threshold: float) -> Optional[int]:
        """
        주어진 feature에 대해 후보들 중 가장 유사한 글로벌 ID 반환

        :param feature: 입력 feature (1D 벡터)
        :param candidates: {global_id: feature_vector} 딕셔너리
        :param threshold: cosine 유사도 threshold (낮을수록 더 유사)
        :return: 가장 유사한 글로벌 ID 또는 None
        """
        if not candidates:
            logger.info("❌ 후보가 없습니다.")
            return None

        ids = list(candidates.keys())
        vectors = np.array(list(candidates.values()))

        logger.info(f"🔍 {len(candidates)}개 후보에 대해 매칭 시작")

        dists = cdist([feature], vectors, metric='cosine')[0]
        min_index = np.argmin(dists)
        min_dist = dists[min_index]

        # 모든 후보의 거리 정보 로깅
        if self.enable_detailed_logging:
            logger.info("📈 모든 후보와의 거리:")
            for i, (global_id, distance) in enumerate(zip(ids, dists)):
                logger.info(f"  ID {global_id}: distance={distance:.4f}")

        logger.info(f"🏆 최적 매치: ID {ids[min_index]} (distance: {min_dist:.4f}, threshold: {threshold:.4f})")

        if min_dist < threshold:
            logger.info(f"✅ 매치 성공: ID {ids[min_index]} (distance: {min_dist:.4f} < threshold: {threshold:.4f})")
            return ids[min_index]
        
        logger.info(f"❌ 매치 실패: 최소 거리 {min_dist:.4f} >= threshold {threshold:.4f}")
        return None

    def calculate_similarity(self, feature1: np.ndarray, feature2: np.ndarray, 
                           context: str = "unknown") -> float:
        """
        두 feature 벡터 간의 cosine 유사도 계산
        
        :param feature1: 첫 번째 feature 벡터
        :param feature2: 두 번째 feature 벡터
        :param context: 계산 컨텍스트 (디버깅용)
        :return: cosine 유사도 (0~1, 높을수록 유사)
        """
        
        # 입력 검증
        if feature1 is None or feature2 is None:
            logger.error(f"❌ {context}: feature가 None입니다.")
            return 0.0
        
        # numpy array로 변환
        if not isinstance(feature1, np.ndarray):
            feature1 = np.array(feature1)
        if not isinstance(feature2, np.ndarray):
            feature2 = np.array(feature2)
        
        # 빈 배열 확인
        if feature1.size == 0 or feature2.size == 0:
            logger.error(f"❌ {context}: feature가 비어있습니다.")
            return 0.0
        
        # 입력 차원 확인 및 정규화
        feature1_flat = feature1.flatten()  # 1차원으로 평탄화
        feature2_flat = feature2.flatten()  # 1차원으로 평탄화
    
        # 차원 확인
        if feature1_flat.shape != feature2_flat.shape:
            logger.error(f"❌ {context}: 차원 불일치 - feature1: {feature1_flat.shape}, feature2: {feature2_flat.shape}")
            return 0.0
        
        # 정규화
        feature1_norm = feature1_flat / np.linalg.norm(feature1_flat)
        feature2_norm = feature2_flat / np.linalg.norm(feature2_flat)
        
        # cosine 유사도 계산
        similarity = 1 - cdist([feature1_norm], [feature2_norm], metric='cosine')[0][0]
        
        # 유사도 히스토리에 저장
        self.similarity_history.append({
            'context': context,
            'feature1_shape': feature1_flat.shape,
            'feature2_shape': feature2_flat.shape,
            'similarity': similarity,
            'timestamp': np.datetime64('now')
        })
        
        
        return similarity

    def get_similarity_history(self, context: str = None) -> list:
        """유사도 계산 히스토리 반환"""
        if context:
            return [h for h in self.similarity_history if h['context'] == context]
        return self.similarity_history

    def clear_history(self):
        """유사도 계산 히스토리 초기화"""
        self.similarity_history.clear()
        logger.info("🗑️ 유사도 계산 히스토리가 초기화되었습니다.")

    def print_summary(self):
        """유사도 계산 요약 출력"""
        if not self.similarity_history:
            logger.info("📊 유사도 계산 히스토리가 없습니다.")
            return
        
        logger.info("📊 유사도 계산 요약:")
        contexts = {}
        for h in self.similarity_history:
            ctx = h['context']
            if ctx not in contexts:
                contexts[ctx] = []
            contexts[ctx].append(h['similarity'])
        
        for ctx, similarities in contexts.items():
            avg_sim = np.mean(similarities)
            min_sim = np.min(similarities)
            max_sim = np.max(similarities)
            logger.info(f"  {ctx}: 평균={avg_sim:.4f}, 최소={min_sim:.4f}, 최대={max_sim:.4f} (총 {len(similarities)}회)")