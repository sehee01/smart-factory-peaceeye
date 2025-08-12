import numpy as np
import redis
from typing import Optional
from ..pre_registration import PreRegistrationManager
from ..similarity import FeatureSimilarityCalculator


class PreRegistrationMatcher:
    """
    사전 등록된 이미지들과 현재 탐지된 객체를 매칭하는 클래스
    """
    
    def __init__(self, pre_reg_manager: PreRegistrationManager, 
                 similarity_calc: FeatureSimilarityCalculator):
        self.pre_reg_manager = pre_reg_manager
        self.similarity = similarity_calc
    
    def match(self, current_feature: np.ndarray) -> Optional[int]:
        """
        현재 feature와 사전 등록된 모든 Global ID와 비교하여 매칭
        
        매칭 로직:
        1. 각 Global ID의 10개 feature와 유사도 계산
        2. 유사도가 0.9 이상인 feature 개수 확인
        3. 2개 이상이면 해당 Global ID 반환
        4. 여러 ID가 조건을 만족하면 더 많은 개수의 feature가 임계값을 넘는 ID 선택
        
        Args:
            current_feature: 현재 탐지된 객체의 feature 벡터
            
        Returns:
            매칭된 Global ID 또는 None
        """
        try:
            # 1. 사전 등록된 모든 Global ID 조회
            pre_registered_ids = self.pre_reg_manager.list_all_pre_registered()
            
            if not pre_registered_ids:
                print("[PreRegistrationMatcher] 사전 등록된 Global ID가 없습니다.")
                return None
            
            print(f"[PreRegistrationMatcher] 사전 등록된 Global ID {len(pre_registered_ids)}개와 매칭 시도")
            
            # 2. 각 Global ID와 유사도 계산
            best_match_id = None
            best_match_count = 0
            
            for global_id in pre_registered_ids:
                try:
                    # 사전 등록된 features 조회
                    pre_features = self.pre_reg_manager.get_pre_registered_features(global_id)
                    
                    if not pre_features or len(pre_features) != 10:
                        print(f"[PreRegistrationMatcher] Global ID {global_id}: feature 개수 부족 ({len(pre_features) if pre_features else 0})")
                        continue
                    
                    # 3. 10개 feature와 유사도 계산
                    match_count = 0
                    for pre_feature in pre_features:
                        try:
                            similarity = self.similarity.calculate_similarity(current_feature, pre_feature)
                            
                            # 유사도가 0.9 이상이면 매칭으로 카운트
                            if similarity >= 0.9:
                                match_count += 1
                                
                        except Exception as e:
                            print(f"[PreRegistrationMatcher] Global ID {global_id} feature 유사도 계산 실패: {str(e)}")
                            continue
                    
                    print(f"[PreRegistrationMatcher] Global ID {global_id}: {match_count}/10 feature 매칭 (임계값: 0.9)")
                    
                    # 4. 2개 이상 매칭되고, 더 많은 개수가 매칭된 경우 선택
                    if match_count >= 2 and match_count > best_match_count:
                        best_match_count = match_count
                        best_match_id = global_id
                        print(f"[PreRegistrationMatcher] 새로운 최고 매칭: Global ID {global_id} ({match_count}/10)")
                        
                except Exception as e:
                    print(f"[PreRegistrationMatcher] Global ID {global_id} 처리 중 오류: {str(e)}")
                    continue
            
            # 5. 매칭 결과 처리
            if best_match_id is not None:
                print(f"[PreRegistrationMatcher] 최종 매칭 성공: Global ID {best_match_id} ({best_match_count}/10 feature)")
                return best_match_id
            else:
                print("[PreRegistrationMatcher] 매칭 조건을 만족하는 Global ID가 없습니다.")
                return None
                
        except redis.ConnectionError:
            print("[PreRegistrationMatcher] Redis 연결 실패")
            return None
        except Exception as e:
            print(f"[PreRegistrationMatcher] 예상치 못한 오류: {str(e)}")
            return None
