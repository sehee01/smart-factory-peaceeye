import redis
import pickle

def quick_check():
    """Redis 상태 빠른 확인"""
    try:
        # Redis 연결
        r = redis.Redis(host='localhost', port=6379, decode_responses=False)
        
        print("=== Redis Global ReID 빠른 확인 ===")
        
        # 모든 키 조회
        all_keys = r.keys("*")
        print(f"총 키 개수: {len(all_keys)}")
        
        if len(all_keys) == 0:
            print("Redis에 데이터가 없습니다.")
            return
        
        # 글로벌 트랙 관련 키들
        track_keys = r.keys("global_track:*")
        feature_keys = r.keys("global_track_features:*")
        metadata_keys = r.keys("global_track_metadata:*")
        
        print(f"트랙 키: {len(track_keys)}개")
        print(f"특징 키: {len(feature_keys)}개")
        print(f"메타데이터 키: {len(metadata_keys)}개")
        
        # 트랙 ID들 출력
        if track_keys:
            print("\n활성 트랙 ID들:")
            for key in track_keys:
                track_id = key.decode().split(':')[1]
                print(f"  - {track_id}")
        
        # 최근 트랙의 상세 정보
        if metadata_keys:
            print("\n최근 트랙 상세 정보:")
            for key in metadata_keys[-3:]:  # 최근 3개만
                track_id = key.decode().split(':')[1]
                metadata = r.get(key)
                if metadata:
                    metadata_dict = pickle.loads(metadata)
                    print(f"  트랙 {track_id}:")
                    print(f"    카메라: {metadata_dict.get('camera_id', 'N/A')}")
                    print(f"    마지막 프레임: {metadata_dict.get('last_seen', 'N/A')}")
                    print(f"    특징 개수: {metadata_dict.get('feature_count', 'N/A')}")
        
    except redis.ConnectionError:
        print("Redis 서버에 연결할 수 없습니다.")
        print("Redis 서버가 실행 중인지 확인하세요.")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    quick_check() 