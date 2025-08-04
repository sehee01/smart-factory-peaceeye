import asyncio
from datetime import datetime, timezone
import json
import requests
import sys
import os
import torch

# 상위 디렉토리 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ByteTrack'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'deep-person-reid-master'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'mapping'))  # point_transformer 경로 추가

# Node.js 서버의 POST 수신 URL
NODE_SERVER_URL = "http://localhost:5000/inference"

# Redis Global ReID 모듈 import
try:
    from redis_global_reid_main_v2 import run_tracking_realtime
    import argparse
except ImportError as e:
    print(f"Redis Global ReID 모듈 import 실패: {e}")
    sys.exit(1)

# ReID 추적 실행을 위한 인자 설정
parser = argparse.ArgumentParser()
parser.add_argument('--videos', nargs='+', type=str, 
                   default=["../test_video/KSEB03.mp4"], 
                   help='List of video file paths.')
parser.add_argument('--yolo_model', type=str, 
                   default="weights/bestcctv.pt", 
                   help='Path to the YOLOv11 model file.')
parser.add_argument('--redis_host', type=str, default="localhost", help='Redis server host.')
parser.add_argument('--redis_port', type=int, default=6379, help='Redis server port.')

args = parser.parse_args([])  # 빈 리스트로 기본값 사용

# ReID 모델 초기화 (한 번만)
try:
    from redis_global_reid_main_v2 import FeatureExtractor, RedisGlobalReIDManagerV2
    reid_extractor = FeatureExtractor(
        model_name='osnet_ibn_x1_0',
        model_path=None,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    global_reid_manager = RedisGlobalReIDManagerV2(
        similarity_threshold=0.5,
        feature_ttl=3000,
        max_features_per_camera=10,
        redis_host='localhost',
        redis_port=6379,
        frame_rate=30
    )
    
    # 실시간 추적 제너레이터 생성
    tracking_generators = []
    for i, video_path in enumerate(args.videos):
        generator = run_tracking_realtime(
            video_path, 
            args.yolo_model, 
            reid_extractor, 
            camera_id=i, 
            global_reid_manager=global_reid_manager
        )
        tracking_generators.append(generator)
    
    print(f"[INFO] Initialized {len(tracking_generators)} video trackers")
    
except Exception as e:
    print(f"ReID 모델 초기화 실패: {e}")
    sys.exit(1)

def run_detection():
    """실시간 ReID 추적에서 현재 프레임 결과를 반환"""
    now = datetime.now(timezone.utc).isoformat()
    
    try:
        # 모든 카메라의 현재 프레임 결과 수집
        all_detections = []
        for generator in tracking_generators:
            try:
                detections = next(generator)  # 다음 프레임 결과 가져오기
                all_detections.extend(detections)
            except StopIteration:
                # 비디오가 끝나면 다시 시작
                continue
        
        # 감지 결과를 workers 형태로 변환
        workers = []
        for detection in all_detections:
            worker = {
                "worker_id": f"worker_{detection['workerID']:03d}",
                "x": float(detection['position_X']),  # float64 유지
                "y": float(detection['position_Y']),  # float64 유지
                "status": "normal",
                "zone_id": f"Z{detection['cameraID']:02d}",
                "product_count": 1,
                "timestamp": now,
                "frame_id": detection.get('frame_id', 0)
            }
            workers.append(worker)
        
        # 알림 생성 (현재는 기본값)
        alerts = []
        
        # 구역 통계 계산
        zone_stats = {}
        for w in workers:
            zid = w["zone_id"]
            zone_stats.setdefault(zid, {"count": 0})
            zone_stats[zid]["count"] += w.get("product_count", 0)

        zones = []
        for zid, stat in zone_stats.items():
            total = stat["count"]
            avg = 480 / total if total > 0 else None
            zones.append({
                "zone_id": zid,
                "zone_name": f"Zone {zid}",
                "zone_type": "작업구역",
                "active_workers": sum(1 for w in workers if w["zone_id"] == zid),
                "active_tasks": "",
                "avg_cycle_time_min": avg,
                "ppe_violations": sum(1 for w in workers if w["zone_id"] == zid and w["status"] == "warning"),
                "hazard_dwell_count": sum(1 for w in workers if w["zone_id"] == zid and w["status"] == "roi_violation"),
                "recent_alerts": ""
            })

        return {
            "timestamp": now,
            "workers": workers,
            "alerts": alerts,
            "zones": zones
        }
        
    except Exception as e:
        print(f"ReID 추적 실행 중 오류: {e}")
        # 오류 발생 시 기본 데이터 반환
        return {
            "timestamp": now,
            "workers": [],
            "alerts": [],
            "zones": []
        }

# 매 프레임 실시간 감지 및 전송 루프
async def detection_loop():
    print("[INFO] AI detection started - Real-time mode")
    while True:
        try:
            result = run_detection()
            if result and result.get('workers'):  # 감지 결과가 있을 때만 전송
                res = requests.post(NODE_SERVER_URL, json=result, timeout=1)
                print(f"[POST] Frame sent: {res.status_code} - {len(result['workers'])} workers")
            else:
                print("[INFO] No workers in frame")
        except requests.RequestException as e:
            print(f"[ERROR] Failed to send to Node.js: {e}")
        except Exception as e:
            print(f"[ERROR] Detection error: {e}")

        await asyncio.sleep(0.033)  # 0.033초 간격 (약 30fps)

if __name__ == "__main__":
    try:
        asyncio.run(detection_loop())
    except Exception as e:
        print(f"[ERROR] {e}")
