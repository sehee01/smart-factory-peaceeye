const express = require("express");
const path = require("path");
const app = express();
const cookieParser = require("cookie-parser");
const cors = require("cors");
require("dotenv").config();

const WebSocket = require("ws"); 
const saveToSQLite = require("./services/saveToSQLite");

// 미들웨어
app.use(cors({
  origin: "http://localhost:3000",  // React 앱 주소
  credentials: true                 // 쿠키 포함 허용
}));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(cookieParser());

// SQLite 초기화
require("./models/initDB"); 

// 라우터 연결 (SQLite 기반 컨트롤러를 내부에서 사용)
app.use("/", require("./routes/loginRoutes"));
app.use("/zones", require("./routes/zoneRoutes"));
app.use("/alerts", require("./routes/alertRoutes"));
app.use("/team", require("./routes/teamRoutes"));

// workers 위치/상태 정보
app.post("/workers", (req, res) => {
  const data = req.body;
  console.log("[POST] /workers\n", JSON.stringify(data, null, 2));

  // DB 저장
  saveToSQLite(data);

  // WebSocket으로 Unity에 브로드캐스트
  wss.clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify({
        type: "workers_update",
        payload: data
      }));
    }
  });

  res.sendStatus(200);
});

//zone 통계 정보
app.post("/zones", (req, res) => {
  const data = req.body;
  console.log("[POST] /zones\n", JSON.stringify(data, null, 2));

  saveToSQLite(data);

  wss.clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify({
        type: "zone_update",
        payload: data
      }));
    }
  });

  res.sendStatus(200);
});

// violations (ppe, roi) 정보
app.post("/violations", (req, res) => {
  const data = req.body;
  console.log("[POST] /violations\n", JSON.stringify(data, null, 2));

  saveToSQLite(data);

  wss.clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify({
        type: "violation_update",
        payload: data
      }));
    }
  });

  res.sendStatus(200);
});

// WebSocket 서버 (Unity WebGL 연결용)
const WS_PORT = process.env.WS_PORT;
const wss = new WebSocket.Server({ port: WS_PORT });
wss.on("connection", (ws) => {
  console.log("[WebSocket] Unity 연결됨");

  ws.on("close", () => {
    console.log("[WebSocket] Unity 연결 종료됨");
  });
});

// React 빌드 결과물 정적 파일로 서빙
app.use(express.static(path.join(__dirname, "frontend/build")));  // client 폴더에 React 있는 경우

// 모든 GET 요청은 React index.html 반환 (정규식 사용)
app.get(/^\/(?!api).*/, (req, res) => {
  res.sendFile(path.join(__dirname, "frontend", "build", "index.html"));
});

// 서버 시작
const SERVER_PORT = process.env.SERVER_PORT;
app.listen(SERVER_PORT, () => {
  console.log(`${SERVER_PORT}번 포트에서 서버 실행 중`);
});
