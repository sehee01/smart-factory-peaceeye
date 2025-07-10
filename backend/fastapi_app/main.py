# backend/fastapi_app/main.py
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket 예시
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("📡 WebSocket client connected")
    while True:
        await ws.send_json({"worker_id": "W-001", "event_type": "danger_enter"})

@app.get("/")
def read_root():
    return {"message": "FastAPI is running!"}