import asyncio
from datetime import datetime, timezone
import requests

# Node.js 서버 URL
WORKER_URL = "http://localhost:5000/workers"
ZONE_URL = "http://localhost:5000/zones"
VIOLATION_URL = "http://localhost:5000/violations"

def run_detection():
    now = datetime.now(timezone.utc).isoformat()

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
