import asyncio
from datetime import datetime, timezone
import json
import requests

# Node.js 서버의 POST 수신 URL
NODE_SERVER_URL = "http://localhost:5000/inference"

# 예시 추론 함수 (실제 모델로 교체 가능)
def run_detection():
    now = datetime.now(timezone.utc).isoformat()

    workers = [
        {"worker_id": "worker_001", "x": 1, "y": 3, "status": "normal", "zone_id": "Z01", "product_count": 2, "timestamp": now},
        {"worker_id": "worker_002", "x": 15, "y": 35, "status": "warning", "zone_id": "Z01", "product_count": 3, "timestamp": now},
        {"worker_id": "worker_003", "x": 20, "y": 10, "status": "roi_violation", "zone_id": "Z02", "product_count": 1, "timestamp": now}
    ]

    alerts = []
    for w in workers:
        if w["status"] == "warning":
            alerts.append({"worker_id": w["worker_id"], "zone_id": w["zone_id"], "type": "ppe_violation", "timestamp": now})
        elif w["status"] == "roi_violation":
            alerts.append({"worker_id": w["worker_id"], "zone_id": w["zone_id"], "type": "roi_violation", "timestamp": now})

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

# 주기적 감지 및 전송 루프
async def detection_loop():
    print("[INFO] AI detection started")
    while True:
        result = run_detection()
        try:
            res = requests.post(NODE_SERVER_URL, json=result, timeout=2)
            print(f"[POST] Sent to Node.js: {res.status_code}")
        except requests.RequestException as e:
            print(f"[ERROR] Failed to send to Node.js: {e}")

        await asyncio.sleep(0.5)

if __name__ == "__main__":
    try:
        asyncio.run(detection_loop())
    except Exception as e:
        print(f"[ERROR] {e}")
