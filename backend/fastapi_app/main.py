# backend/fastapi_app/main.py
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import socketio
from dotenv import load_dotenv

load_dotenv()


# 🧠 Socket.IO 서버 생성
sio = socketio.AsyncServer(async_mode="asgi")
fastapi_app = FastAPI()  # ✅ FastAPI 인스턴스

# 최종적으로 통합된 앱
app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)

# 🔽 현재 파일 위치 기준으로 정확한 경로 계산
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUILD_DIR = os.path.join(BASE_DIR, "admin-dashboard", "build")

# CORS 미들웨어 설정
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("CORS_ORIGIN")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket 예시
@fastapi_app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("📡 WebSocket client connected")
    while True:
        await ws.send_json({"worker_id": "W-001", "event_type": "danger_enter"})

# 📁 정적 파일 서빙
fastapi_app.mount("/static", StaticFiles(directory=os.path.join(BUILD_DIR, "static")), name="static")

# 🏠 루트에서 index.html 반환
@fastapi_app.get("/")
def read_root():
    return FileResponse(os.path.join(BUILD_DIR, "index.html"))

# 🎉 Socket.IO 이벤트 예시
@sio.event
async def connect(sid, environ):
    print("✅ 클라이언트 연결됨:", sid)

@sio.event
async def disconnect(sid):
    print("❌ 클라이언트 연결 해제:", sid)
