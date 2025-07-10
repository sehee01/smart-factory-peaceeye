require('dotenv').config();

const express = require('express');
const http = require('http');
const { Server } = require('socket.io');

const app = express();
const server = http.createServer(app);

const PORT = process.env.PORT || 3002;
const ORIGIN = process.env.CORS_ORIGIN || '*';

const io = new Server(server, {
  cors: {
    origin: ORIGIN,
    methods: ["GET", "POST"]
  }
});

// ✅ 클라이언트 연결 시
io.on('connection', (socket) => {
  console.log(`👋 클라이언트 연결됨: ${socket.id}`);

  // ✅ 예시 이벤트 주기적으로 전송
  const interval = setInterval(() => {
    const fakeEvent = {
      worker_id: `W-00${Math.floor(Math.random() * 4 + 1)}`,
      event_type: 'Entered danger zone',
      timestamp: new Date().toISOString()
    };

    socket.emit('workerEvent', fakeEvent);
  }, 3000);

  // ✅ 연결 종료 시 정리
  socket.on('disconnect', () => {
    console.log(`❌ 클라이언트 연결 해제: ${socket.id}`);
    clearInterval(interval);
  });
});

server.listen(3002, () => {
  console.log('🚀 WebSocket 서버 포트 3002에서 실행 중');
});