import redis
import pickle
import json
import numpy as np
from datetime import datetime

class RedisMonitorV2:
    """Redis Global ReID V2 모니터링 도구 (통합된 구조)"""
    
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
    
    def get_all_tracks(self):
        """모든 트랙 정보 조회 (통합된 구조)"""
        tracks = {}
        
        track_keys = self.redis_client.keys("global_track:*")
        for track_key in track_keys:
            track_id = track_key.decode().split(':')[1]
            tracks[track_id] = self.get_track_details(track_id)
        
        return tracks
    
    def get_track_details(self, track_id):
        """트랙 상세 정보 조회 (통합된 구조)"""
        data_key = f"global_track_data:{track_id}"
        track_data = self.redis_client.get(data_key)
        
        if track_data:
            track_info = pickle.loads(track_data)
            return {
                'track_id': track_id,
                'is_disappeared': track_info.get('is_disappeared', False),
                'disappeared_since': track_info.get('disappeared_since'),
                'last_activity': track_info.get('last_activity'),
                'cameras': track_info['cameras'],
                'total_cameras': len(track_info['cameras']),
                'total_features': sum(len(cam_data['features']) for cam_data in track_info['cameras'].values())
            }
        return None
    
    def get_redis_stats(self):
        """Redis 통계 정보 (통합된 구조)"""
        track_keys = self.redis_client.keys("global_track:*")
        tracks = {}
        
        # 상태별 통계 계산
        normal_count = 0
        disappeared_count = 0
        
        for track_key in track_keys:
            track_id = track_key.decode().split(':')[1]
            track_info = self.get_track_details(track_id)
            if track_info:
                if track_info['is_disappeared']:
                    disappeared_count += 1
                else:
                    normal_count += 1
        
        stats = {
            'total_tracks': len(track_keys),
            'normal_tracks': normal_count,
            'disappeared_tracks': disappeared_count,
            'total_keys': len(self.redis_client.keys("*")),
            'memory_usage': self.redis_client.info()['used_memory_human']
        }
        return stats
    
    def clear_all_tracks(self):
        """모든 트랙 삭제 (통합된 구조)"""
        track_keys = self.redis_client.keys("global_track:*")
        track_data_keys = self.redis_client.keys("global_track_data:*")
        history_keys = self.redis_client.keys("track_history:*")
        
        all_keys = track_keys + track_data_keys + history_keys
        
        if all_keys:
            self.redis_client.delete(*all_keys)
            print(f"Cleared {len(all_keys)} keys from Redis")
        else:
            print("No tracks to clear")
    
    def monitor_realtime(self, interval=5):
        """실시간 모니터링 (통합된 구조)"""
        import time
        
        print("Real-time Redis Global ReID V2 Monitor (Unified Structure)")
        print("Press Ctrl+C to stop")
        print("-" * 50)
        
        try:
            while True:
                stats = self.get_redis_stats()
                print(f"\n[{time.strftime('%H:%M:%S')}] Redis Stats:")
                print(f"  Total Tracks: {stats['total_tracks']}")
                print(f"  Normal Tracks: {stats['normal_tracks']}")
                print(f"  Disappeared Tracks: {stats['disappeared_tracks']}")
                print(f"  Total Keys: {stats['total_keys']}")
                print(f"  Memory Usage: {stats['memory_usage']}")
                
                # 최근 트랙들 표시
                tracks = self.get_all_tracks()
                if tracks:
                    print(f"\nRecent Tracks ({len(tracks)} total):")
                    for track_id, track_info in list(tracks.items())[:5]:  # 최근 5개만
                        if track_info:
                            status = "DISAPPEARED" if track_info['is_disappeared'] else "NORMAL"
                            print(f"  {track_id} ({status}) - Cameras: {track_info['total_cameras']}, Features: {track_info['total_features']}")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
    
    def get_cross_camera_tracks(self):
        """여러 카메라에 걸친 트랙들 조회 (통합된 구조)"""
        cross_camera_tracks = {}
        
        track_keys = self.redis_client.keys("global_track:*")
        for track_key in track_keys:
            track_id = track_key.decode().split(':')[1]
            track_info = self.get_track_details(track_id)
            if track_info and track_info['total_cameras'] > 1:
                cross_camera_tracks[track_id] = track_info
        
        return cross_camera_tracks

def main():
    monitor = RedisMonitorV2()
    
    while True:
        print("\n=== Redis Global ReID V2 모니터 ===")
        print("1. 모든 트랙 정보 조회")
        print("2. 특정 트랙 상세 정보")
        print("3. Redis 통계 정보")
        print("4. 실시간 모니터링")
        print("5. 다중 카메라 연결 트랙 조회")
        print("6. 모든 데이터 삭제")
        print("7. 종료")
        
        choice = input("\n선택하세요 (1-7): ")
        
        if choice == '1':
            tracks = monitor.get_all_tracks()
            if tracks:
                print("\n=== 모든 트랙 정보 ===")
                for track_id, track_info in tracks.items():
                    if track_info:
                        status = "DISAPPEARED" if track_info['is_disappeared'] else "NORMAL"
                        print(f"\n--- 트랙 ID: {track_id} ({status}) ---")
                        print(f"  총 카메라: {track_info['total_cameras']}")
                        print(f"  총 특징: {track_info['total_features']}")
                        print(f"  연결된 카메라: {list(track_info['cameras'].keys())}")
                    else:
                        print(f"  트랙 ID: {track_id} (데이터 없음)")
            else:
                print("저장된 트랙이 없습니다.")
        elif choice == '2':
            track_id = input("트랙 ID를 입력하세요: ")
            track_info = monitor.get_track_details(track_id)
            if track_info:
                print(f"\n--- 트랙 ID: {track_id} (NORMAL) ---")
                print(f"  총 카메라: {track_info['total_cameras']}")
                print(f"  총 특징: {track_info['total_features']}")
                print(f"  연결된 카메라: {list(track_info['cameras'].keys())}")
            else:
                print(f"트랙 ID: {track_id} (데이터 없음)")
        elif choice == '3':
            stats = monitor.get_redis_stats()
            print("\n=== Redis 통계 정보 ===")
            print(f"총 키 개수: {stats['total_keys']}")
            print(f"활성 글로벌 트랙: {stats['total_tracks']}")
            print(f"사라진 트랙: {stats['disappeared_tracks']}")
            print(f"사용된 메모리: {stats['memory_usage']}")
        elif choice == '4':
            interval = input("모니터링 간격(초)을 입력하세요 (기본: 5): ")
            try:
                interval = int(interval) if interval else 5
                monitor.monitor_realtime(interval)
            except ValueError:
                print("올바른 숫자를 입력하세요.")
        elif choice == '5':
            cross_camera_tracks = monitor.get_cross_camera_tracks()
            if cross_camera_tracks:
                print("\n=== 다중 카메라 연결 트랙들 ===")
                for track_id, track_info in cross_camera_tracks.items():
                    status = "DISAPPEARED" if track_info['is_disappeared'] else "NORMAL"
                    print(f"\n--- 트랙 ID: {track_id} ({status}) ---")
                    print(f"  연결된 카메라: {list(track_info['cameras'].keys())}")
                    print(f"  총 특징: {track_info['total_features']}")
            else:
                print("다중 카메라에 연결된 트랙이 없습니다.")
        elif choice == '6':
            confirm = input("정말 모든 데이터를 삭제하시겠습니까? (y/N): ")
            if confirm.lower() == 'y':
                monitor.clear_all_tracks()
        elif choice == '7':
            print("종료합니다.")
            break
        else:
            print("올바른 선택을 해주세요.")

if __name__ == "__main__":
    main() 