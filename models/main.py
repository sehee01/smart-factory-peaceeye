import os
import asyncio
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime, timezone
import json
import certifi
import websockets

# .env 파일 로딩
load_dotenv()

# 환경 변수에서 URI 불러오기
mongo_uri = os.environ.get("MONGO_URI")
client = MongoClient(mongo_uri, tlsCAFile=certifi.where())

# DB 선택
db = client["peaceeye"]
collection = db["detections"]

# 접속한 WebSocket 클라이언트 저장용
connected_clients = set()

# 예시 추론 함수 (실제 모델로 교체 가능)
def run_detection():
    return {
        "workers": [
            {"worker_id": "worker_001", "x": 100, "y": 100, "status": "normal"},
            {"worker_id": "worker_002", "x": 30, "y": 15, "status": "warning"},
        ]
    }

# WebSocket 클라이언트 연결 핸들러
async def websocket_handler(websocket):
    print("[INFO] Unity client connected.")
    connected_clients.add(websocket)

    try:
        while True:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                print(f"[INFO] Message from Unity: {message}")
                # 메시지를 받은 후 원하는 동작 트리거 가능 (예: 감지 루프 시작 등)
            except asyncio.TimeoutError:
                continue
            except websockets.ConnectionClosed:
                print("[INFO] Unity client disconnected.")
                break
    finally:
        connected_clients.remove(websocket)

# 주기적 감지 및 전송 루프
async def detection_loop():
    print("[INFO] Starting AI detection loop...")

    while True:
        result = run_detection()
        # MongoDB에는 timestamp 포함해서 저장
        db_result = {
            "timestamp": datetime.now(timezone.utc),
            **result
        }
        collection.insert_one(db_result)
        print(f"[INFO] Saved detection at {db_result['timestamp']}")

        # WebSocket 클라이언트에게 전송
        if connected_clients:
            message = json.dumps(result, default=str)
            disconnected = set()

            for client in connected_clients:
                try:
                    await client.send(message)
                except Exception as e:
                    print(f"[ERROR] Failed to send to client: {e}")
                    disconnected.add(client)

            connected_clients.difference_update(disconnected)

        await asyncio.sleep(0.5)

# 메인 실행
async def main():
    # WebSocket 서버 시작
    server = await websockets.serve(websocket_handler, "0.0.0.0", 8000)
    print("[INFO] WebSocket server running at ws://localhost:8000")

    # 감지 루프 실행
    await detection_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"[ERROR] {e}")
