import redis
import pickle
import json
import numpy as np
from datetime import datetime

class RedisMonitor:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
    
    def get_all_tracks(self):
        """모든 글로벌 트랙 정보 조회"""
        print("=== 모든 글로벌 트랙 정보 ===")
        
        # 모든 트랙 키 조회
        track_keys = self.redis_client.keys("global_track:*")
        
        if not track_keys:
            print("저장된 트랙이 없습니다.")
            return
        
        print(f"총 {len(track_keys)}개의 트랙이 있습니다.\n")
        
        for track_key in track_keys:
            track_id = track_key.decode().split(':')[1]
            print(f"--- 트랙 ID: {track_id} ---")
            
            # 메타데이터 조회
            metadata_key = f"global_track_metadata:{track_id}"
            metadata = self.redis_client.get(metadata_key)
            
            if metadata:
                metadata_dict = pickle.loads(metadata)
                print(f"카메라 ID: {metadata_dict.get('camera_id', 'N/A')}")
                print(f"마지막 업데이트: {metadata_dict.get('last_seen', 'N/A')}")
                print(f"특징 개수: {metadata_dict.get('feature_count', 'N/A')}")
                print(f"마지막 위치: {metadata_dict.get('last_bbox', 'N/A')}")
                
                # TTL 확인
                ttl = self.redis_client.ttl(metadata_key)
                print(f"TTL (초): {ttl}")
            else:
                print("메타데이터가 없습니다.")
            
            # 특징 정보 조회
            feature_key = f"global_track_features:{track_id}"
            features_data = self.redis_client.get(feature_key)
            
            if features_data:
                features = pickle.loads(features_data)
                print(f"저장된 특징 벡터 개수: {len(features)}")
                if len(features) > 0:
                    print(f"특징 벡터 차원: {features[0].shape}")
                    print(f"첫 번째 특징 벡터 (처음 5개 값): {features[0][:5]}")
            else:
                print("특징 데이터가 없습니다.")
            
            print()
    
    def get_track_details(self, track_id):
        """특정 트랙의 상세 정보 조회"""
        print(f"=== 트랙 {track_id} 상세 정보 ===")
        
        # 메타데이터 조회
        metadata_key = f"global_track_metadata:{track_id}"
        metadata = self.redis_client.get(metadata_key)
        
        if metadata:
            metadata_dict = pickle.loads(metadata)
            print("메타데이터:")
            print(json.dumps(metadata_dict, indent=2, default=str))
        else:
            print("메타데이터가 없습니다.")
        
        # 특징 데이터 조회
        feature_key = f"global_track_features:{track_id}"
        features_data = self.redis_client.get(feature_key)
        
        if features_data:
            features = pickle.loads(features_data)
            print(f"\n특징 데이터:")
            print(f"특징 벡터 개수: {len(features)}")
            for i, feature in enumerate(features):
                print(f"  특징 {i+1}: shape={feature.shape}, norm={np.linalg.norm(feature):.4f}")
        else:
            print("특징 데이터가 없습니다.")
    
    def get_redis_stats(self):
        """Redis 통계 정보 조회"""
        print("=== Redis 통계 정보 ===")
        
        # 데이터베이스 크기
        db_size = self.redis_client.dbsize()
        print(f"총 키 개수: {db_size}")
        
        # 메모리 사용량
        info = self.redis_client.info('memory')
        print(f"사용된 메모리: {info.get('used_memory_human', 'N/A')}")
        print(f"메모리 피크: {info.get('used_memory_peak_human', 'N/A')}")
        
        # 키 패턴별 개수
        patterns = [
            "global_track:*",
            "global_track_features:*", 
            "global_track_metadata:*"
        ]
        
        print("\n키 패턴별 개수:")
        for pattern in patterns:
            count = len(self.redis_client.keys(pattern))
            print(f"  {pattern}: {count}개")
    
    def clear_all_tracks(self):
        """모든 글로벌 트랙 데이터 삭제"""
        print("=== 모든 글로벌 트랙 데이터 삭제 ===")
        
        track_keys = self.redis_client.keys("global_track:*")
        feature_keys = self.redis_client.keys("global_track_features:*")
        metadata_keys = self.redis_client.keys("global_track_metadata:*")
        
        all_keys = track_keys + feature_keys + metadata_keys
        
        if all_keys:
            deleted = self.redis_client.delete(*all_keys)
            print(f"{deleted}개의 키가 삭제되었습니다.")
        else:
            print("삭제할 데이터가 없습니다.")
    
    def monitor_realtime(self, interval=5):
        """실시간 모니터링"""
        print(f"=== 실시간 모니터링 (간격: {interval}초) ===")
        print("Ctrl+C로 종료")
        
        try:
            while True:
                import time
                time.sleep(interval)
                
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 현재 상태:")
                track_count = len(self.redis_client.keys("global_track:*"))
                print(f"활성 트랙: {track_count}개")
                
                # 최근 업데이트된 트랙들
                if track_count > 0:
                    print("최근 트랙들:")
                    track_keys = self.redis_client.keys("global_track:*")
                    for track_key in track_keys[-3:]:  # 최근 3개만
                        track_id = track_key.decode().split(':')[1]
                        metadata_key = f"global_track_metadata:{track_id}"
                        metadata = self.redis_client.get(metadata_key)
                        if metadata:
                            metadata_dict = pickle.loads(metadata)
                            last_seen = metadata_dict.get('last_seen', 0)
                            camera_id = metadata_dict.get('camera_id', 'N/A')
                            print(f"  ID {track_id}: 카메라 {camera_id}, 프레임 {last_seen}")
                
        except KeyboardInterrupt:
            print("\n모니터링 종료")

def main():
    monitor = RedisMonitor()
    
    while True:
        print("\n=== Redis Global ReID 모니터 ===")
        print("1. 모든 트랙 정보 조회")
        print("2. 특정 트랙 상세 정보")
        print("3. Redis 통계 정보")
        print("4. 실시간 모니터링")
        print("5. 모든 데이터 삭제")
        print("6. 종료")
        
        choice = input("\n선택하세요 (1-6): ")
        
        if choice == '1':
            monitor.get_all_tracks()
        elif choice == '2':
            track_id = input("트랙 ID를 입력하세요: ")
            monitor.get_track_details(track_id)
        elif choice == '3':
            monitor.get_redis_stats()
        elif choice == '4':
            interval = input("모니터링 간격(초)을 입력하세요 (기본: 5): ")
            try:
                interval = int(interval) if interval else 5
                monitor.monitor_realtime(interval)
            except ValueError:
                print("올바른 숫자를 입력하세요.")
        elif choice == '5':
            confirm = input("정말 모든 데이터를 삭제하시겠습니까? (y/N): ")
            if confirm.lower() == 'y':
                monitor.clear_all_tracks()
        elif choice == '6':
            print("종료합니다.")
            break
        else:
            print("올바른 선택을 해주세요.")

if __name__ == "__main__":
    main() 