import numpy as np
import redis
from typing import Optional
from ..pre_registration import PreRegistrationManager
from ..similarity import FeatureSimilarityCalculator
import sys
import os

# app 디렉토리 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(os.path.dirname(current_dir))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from config import settings


class PreRegistrationMatcher:
    """
    사전 등록된 이미지들과 현재 탐지된 객체를 매칭하는 클래스
    """
    
    def __init__(self, pre_reg_manager: PreRegistrationManager, 
                 similarity_calc: FeatureSimilarityCalculator):
        self.pre_reg_manager = pre_reg_manager
        self.similarity = similarity_calc
        
        # settings.py에서 설정 가져오기
        self.similarity_threshold = settings.REID_CONFIG["pre_registration"]["similarity_threshold"]
        self.min_matching_features = settings.REID_CONFIG["pre_registration"]["min_matching_features"]
        self.max_features_per_id = settings.REID_CONFIG["pre_registration"]["max_features_per_id"]
    
    def match(self, current_feature: np.ndarray) -> Optional[int]:
        """
        현재 feature와 사전 등록된 모든 Global ID와 비교하여 매칭
        
        매칭 로직:
        1. 각 Global ID의 10개 feature와 유사도 계산
        2. 유사도가 설정된 임계값 이상인 feature 개수 확인
        3. 최소 매칭 개수 이상이면 해당 Global ID 반환
        4. 여러 ID가 조건을 만족하면 더 많은 개수의 feature가 임계값을 넘는 ID 선택
        
        Args:
            current_feature: 현재 탐지된 객체의 feature 벡터
            
        Returns:
            매칭된 Global ID 또는 None
        """
        try:
            print(f"[DEBUG] === 사전 등록 매칭 시작 ===")
            print(f"[DEBUG] 현재 feature 차원: {current_feature.shape}")
            print(f"[DEBUG] 현재 feature 값 범위: {current_feature.min():.4f} ~ {current_feature.max():.4f}")
            print(f"[DEBUG] 설정값: 임계값={self.similarity_threshold}, 최소매칭={self.min_matching_features}개")
            
            # 1. 사전 등록된 모든 Global ID 조회
            pre_registered_ids = self.pre_reg_manager.list_all_pre_registered()
            
            if not pre_registered_ids:
                print("[DEBUG] ❌ 사전 등록된 Global ID가 없습니다.")
                return None
            
            print(f"[DEBUG] ✅ 사전 등록된 Global ID {len(pre_registered_ids)}개 발견: {pre_registered_ids}")
            
            # 2. 각 Global ID와 유사도 계산
            best_match_id = None
            best_match_count = 0
            
            for global_id in pre_registered_ids:
                print(f"[DEBUG] --- Global ID {global_id} 매칭 시도 ---")
                
                try:
                    # 사전 등록된 features 조회
                    pre_features = self.pre_reg_manager.get_pre_registered_features(global_id)
                    
                    if not pre_features or len(pre_features) != self.max_features_per_id:
                        print(f"[DEBUG] ❌ Global ID {global_id}: feature 개수 부족 ({len(pre_features) if pre_features else 0}/{self.max_features_per_id})")
                        continue
                    
                    print(f"[DEBUG] Global ID {global_id}: {len(pre_features)}개 feature 로드 완료")
                    
                    # 3. feature들과 유사도 계산
                    match_count = 0
                    similarities = []
                    
                    for i, pre_feature in enumerate(pre_features):
                        try:
                            similarity = self.similarity.calculate_similarity(current_feature, pre_feature)
                            similarities.append(similarity)
                            
                            # 유사도가 설정된 임계값 이상이면 매칭으로 카운트
                            if similarity >= self.similarity_threshold:
                                match_count += 1
                                print(f"[DEBUG]   Feature {i+1}: 유사도 {similarity:.4f} >= {self.similarity_threshold} ✅")
                            else:
                                print(f"[DEBUG]   Feature {i+1}: 유사도 {similarity:.4f} < {self.similarity_threshold} ❌")
                                
                        except Exception as e:
                            print(f"[DEBUG] ❌ Global ID {global_id} feature {i+1} 유사도 계산 실패: {str(e)}")
                            continue
                    
                    print(f"[DEBUG] Global ID {global_id}: {match_count}/{self.max_features_per_id} feature 매칭 (임계값: {self.similarity_threshold})")
                    print(f"[DEBUG] 유사도 분포: {[f'{s:.4f}' for s in similarities]}")
                    
                    # 4. 최소 매칭 개수 이상이고, 더 많은 개수가 매칭된 경우 선택
                    if match_count >= self.min_matching_features and match_count > best_match_count:
                        best_match_count = match_count
                        best_match_id = global_id
                        print(f"[DEBUG] 🎯 새로운 최고 매칭: Global ID {global_id} ({match_count}/{self.max_features_per_id})")
                    elif match_count >= self.min_matching_features:
                        print(f"[DEBUG] ⚠️ Global ID {global_id}도 조건 만족하지만 더 낮은 매칭 개수 ({match_count} <= {best_match_count})")
                    else:
                        print(f"[DEBUG] ❌ Global ID {global_id}: 최소 매칭 개수 미달 ({match_count} < {self.min_matching_features})")
                        
                except Exception as e:
                    print(f"[DEBUG] ❌ Global ID {global_id} 처리 중 오류: {str(e)}")
                    continue
            
            # 5. 매칭 결과 처리
            if best_match_id is not None:
                print(f"[DEBUG] 🎉 최종 매칭 성공: Global ID {best_match_id} ({best_match_count}/{self.max_features_per_id} feature)")
                return best_match_id
            else:
                print(f"[DEBUG] ❌ 매칭 조건을 만족하는 Global ID가 없습니다. (최소 {self.min_matching_features}개 feature 필요)")
                return None
                
        except redis.ConnectionError:
            print("[DEBUG] ❌ Redis 연결 실패")
            return None
        except Exception as e:
            print(f"[DEBUG] ❌ 예상치 못한 오류: {str(e)}")
            return None
