import asyncio
from datetime import datetime, timezone
import requests
import json
import sys
import os
import torch
import time

# 프로젝트 루트 경로 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.extend([
    project_root,
    os.path.join(project_root, 'ByteTrack'),
    os.path.join(project_root, 'deep-person-reid-master'),
    os.path.join(project_root, 'models', 'mapping')  # point_transformer 경로
])

# Node.js 서버 URL
WORKER_URL = "http://localhost:5000/workers"
ZONE_URL = "http://localhost:5000/zones"
VIOLATION_URL = "http://localhost:5000/violations"

# Redis Global ReID 모듈 import
try:
    from redis_global_reid_main_v2 import run_tracking_realtime, FeatureExtractor, RedisGlobalReIDManagerV2
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
    start_time = time.time()  # 프레임 처리 시작 시간
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
            
        # 작업자 위치/생산량 정보
        workers = [
            {
                "worker_id": "W001", "x": 1, "y": 3, "zone_id": "Z01",
                "product_count": 2, "timestamp": now
            },
            {
                "worker_id": "W002", "x": 15, "y": 35, "zone_id": "Z01",
                "product_count": 3, "timestamp": now
            },
            {
                "worker_id": "W003", "x": 20, "y": 10, "zone_id": "Z02",
                "product_count": 1, "timestamp": now
            }
        ]

        # 위반 정보 (PPE, ROI)
        violations = [
            {
                "worker_id": "W001",
                "zone_id": "Z01",
                "timestamp": now,
                "violations": {
                    "ppe": ["helmet_missing", "vest_missing"],
                    "roi": []
                }
            },
            {
                "worker_id": "W003",
                "zone_id": "Z02",
                "timestamp": now,
                "violations": {
                    "ppe": [],
                    "roi": ["restricted_area_1"]
                }
            }
        ]

        # zone 통계
        zone_stats = {}
        for w in workers:
            zid = w["zone_id"]
            zone_stats.setdefault(zid, {
                "zone_name": f"Zone {zid}",
                "zone_type": "작업구역",
                "total_product": 0,
                "active_workers": 0
            })
            zone_stats[zid]["total_product"] += w.get("product_count", 0)
            zone_stats[zid]["active_workers"] += 1

        zones = []
        for zid, stat in zone_stats.items():
            total = stat["total_product"]
            avg = 480 / total if total > 0 else None
            zones.append({
                "zone_id": zid,
                "zone_name": stat["zone_name"],
                "zone_type": stat["zone_type"],
                "timestamp": now,
                "active_workers": stat["active_workers"],
                "active_tasks": "",
                "avg_cycle_time_min": avg,
                "ppe_violations": sum(
                    1 for v in violations if v["zone_id"] == zid and v["violations"]["ppe"]
                ),
                "hazard_dwell_count": sum(
                    1 for v in violations if v["zone_id"] == zid and v["violations"]["roi"]
                ),
                "recent_alerts": ""
            })

        return {
            "workers": workers,
            "violations": violations,
            "zones": zones
        }
        
    except Exception as e:
        processing_time = time.time() - start_time  # 오류 시에도 처리 시간 계산
        print(f"ReID 추적 실행 중 오류: {e}")
        # 오류 발생 시 기본 데이터 반환
        return {
            "workers": [],
            "violations": [],
            "zones": [],
        }

# 주기적 전송 루프
async def detection_loop():
    print("[INFO] AI detection loop started")
    while True:
        result = run_detection()

        try:
            # 1. workers
            res1 = requests.post(WORKER_URL, json={"workers": result["workers"]}, timeout=2)
            print(f"[POST] /workers → {res1.status_code}")

            # 2. violations (없으면 건너뜀)
            if result["violations"]:
                res2 = requests.post(VIOLATION_URL, json={"violations": result["violations"]}, timeout=2)
                print(f"[POST] /violations → {res2.status_code}")
            else:
                print("[SKIP] No violations to report.")

            # 3. zones
            res3 = requests.post(ZONE_URL, json={"zones": result["zones"]}, timeout=2)
            print(f"[POST] /zones → {res3.status_code}")

        except requests.RequestException as e:
            print(f"[ERROR] Failed to send: {e}")

        await asyncio.sleep(1.0)

if __name__ == "__main__":
    try:
        asyncio.run(detection_loop())
    except Exception as e:
        print(f"[ERROR] {e}")
