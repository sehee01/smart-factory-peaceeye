import os
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime, timezone
import time
import certifi

# .env 파일 로딩
load_dotenv()

# 환경 변수에서 URI 불러오기
mongo_uri = os.environ.get("MONGO_URI")
client = MongoClient(mongo_uri, tlsCAFile=certifi.where())

# DB 선택
db = client["peaceeye"]
collection = db["detections"]

# 예시 추론 함수 (실제 모델로 교체 가능)
def run_detection():
    return {
        "timestamp": datetime.now(timezone.utc),
        "workers": [
            {"worker_id": "worker_001", "x": 100, "y": 200, "status": "normal"},
            {"worker_id": "worker_002", "x": 300, "y": 150, "status": "warning"},
        ]
    }

# 감지 루프
try:
    print("[INFO] Starting AI detection loop...")

    while True:
        result = run_detection()
        collection.insert_one(result)
        print(f"[INFO] Saved detection at {result['timestamp']}")
        time.sleep(0.5)

except Exception as e:
    print(f"[ERROR] {e}")