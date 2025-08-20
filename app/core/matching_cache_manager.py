from typing import Optional, Dict, Any
import time


class MatchingCacheManager:
    """
    사전 등록 매칭 결과를 캐싱하여 연산량을 줄이는 매니저
    
    캐시 구조:
    - pre_reg_matching_cache: key="camera_id_local_id", value=global_id
    - global_to_local_mapping: key=global_id, value={camera_id: local_id}
    """
    
    def __init__(self):
        # 캐시 저장소
        self.pre_reg_matching_cache = {}  # key: "camera_id_local_id", value: global_id
        self.global_to_local_mapping = {}  # key: global_id, value: {camera_id: local_id}
        
        # 통계
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_queries = 0
    
    def add_matching_cache(self, camera_id: str, local_id: int, global_id: int):
        """
        매칭 캐시 추가 및 중복 제거
        
        Args:
            camera_id: 카메라 ID
            local_id: ByteTrack에서 부여한 local ID
            global_id: 매칭 결과 Global ID
        """
        camera_id = str(camera_id)
        local_id = int(local_id)
        global_id = int(global_id)
        
        # 캐시 키 생성
        cache_key = f"{camera_id}_{local_id}"
        
        # 기존 매핑 확인
        if global_id in self.global_to_local_mapping:
            if camera_id in self.global_to_local_mapping[global_id]:
                # 더 작은 local_id는 삭제
                old_local_id = self.global_to_local_mapping[global_id][camera_id]
                if local_id <= old_local_id:
                    return  # 더 작은 local_id는 무시
                
                # 기존 캐시 항목 삭제
                old_cache_key = f"{camera_id}_{old_local_id}"
                self.pre_reg_matching_cache.pop(old_cache_key, None)
                print(f"[DEBUG] 🗑️ 기존 캐시 항목 삭제: {old_cache_key} (더 큰 local_id로 교체)")
        
        # 새 매핑 추가
        self.pre_reg_matching_cache[cache_key] = global_id
        
        # 역방향 매핑 업데이트
        if global_id not in self.global_to_local_mapping:
            self.global_to_local_mapping[global_id] = {}
        self.global_to_local_mapping[global_id][camera_id] = local_id
        
        print(f"[DEBUG] 💾 캐시 추가: {cache_key} -> Global ID {global_id}")
    
    def get_cached_global_id(self, camera_id: str, local_id: int) -> Optional[int]:
        """
        캐시된 Global ID 조회
        
        Args:
            camera_id: 카메라 ID
            local_id: ByteTrack에서 부여한 local ID
            
        Returns:
            캐시된 Global ID 또는 None (캐시 미스)
        """
        camera_id = str(camera_id)
        local_id = int(local_id)
        
        # 캐시 키 생성
        cache_key = f"{camera_id}_{local_id}"
        
        # 통계 업데이트
        self.total_queries += 1
        
        # 캐시 조회
        if cache_key in self.pre_reg_matching_cache:
            cached_global_id = self.pre_reg_matching_cache[cache_key]
            self.cache_hits += 1
            print(f"[DEBUG] ✅ 캐시 히트: {cache_key} -> Global ID {cached_global_id}")
            return cached_global_id
        else:
            self.cache_misses += 1
            print(f"[DEBUG] ❌ 캐시 미스: {cache_key}")
            return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        캐시 통계 반환
        
        Returns:
            캐시 통계 딕셔너리
        """
        hit_rate = (self.cache_hits / self.total_queries * 100) if self.total_queries > 0 else 0
        
        return {
            "cache_size": len(self.pre_reg_matching_cache),
            "unique_global_ids": len(self.global_to_local_mapping),
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "cache_entries": list(self.pre_reg_matching_cache.items())
        }
    
    def clear_cache(self):
        """캐시 초기화"""
        self.pre_reg_matching_cache.clear()
        self.global_to_local_mapping.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_queries = 0
        print("[DEBUG] 🗑️ 캐시 초기화 완료")
    
    def print_cache_stats(self):
        """캐시 통계 출력"""
        stats = self.get_cache_stats()
        print(f"[CACHE STATS] 크기: {stats['cache_size']}, "
              f"고유 Global ID: {stats['unique_global_ids']}, "
              f"히트율: {stats['hit_rate_percent']}% "
              f"({stats['cache_hits']}/{stats['total_queries']})")
    
    def get_global_id_mapping(self, global_id: int) -> Dict[str, int]:
        """
        특정 Global ID의 카메라별 Local ID 매핑 조회
        
        Args:
            global_id: 조회할 Global ID
            
        Returns:
            {camera_id: local_id} 딕셔너리
        """
        return self.global_to_local_mapping.get(global_id, {})
