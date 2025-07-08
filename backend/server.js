//테스트용 WebSocket 서버
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "*"
  }
});

io.on('connection', socket => {
  console.log('👋 클라이언트 연결됨');

  setInterval(() => {
    const fakeEvent = {
      worker_id: `W-00${Math.floor(Math.random() * 4 + 1)}`,
      event_type: 'Entered danger zone'
    };
    socket.emit('workerEvent', fakeEvent);
  }, 3000);
});

server.listen(3001, () => {
  console.log('🚀 WebSocket 서버 포트 3001에서 실행 중');
});