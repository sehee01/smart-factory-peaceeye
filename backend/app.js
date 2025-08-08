// app.js (완성본)
require("dotenv").config();

const path = require("path");
const express = require("express");
const cors = require("cors");
const cookieParser = require("cookie-parser");
const WebSocket = require("ws");

// ── 포트 ─────────────────────────────────────────────────────────────
const SERVER_PORT = process.env.SERVER_PORT || 5000; // HTTP(REST)
const WS_PORT     = process.env.WS_PORT || 8000;     // WebSocket

// ── 서비스/DB 모듈 ───────────────────────────────────────────────────
const saveToSQLite = require("./services/saveToSQLite"); // Promise 반환 권장
require("./models/initDB"); // SQLite 스키마 초기화

// ── Express ─────────────────────────────────────────────────────────
const app = express();

app.use(cors({
  origin: "http://localhost:3000",  // React 개발 서버
  credentials: true
}));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(cookieParser());

// ── 로그인/도메인 라우트 ────────────────────────
app.use("/",        require("./routes/loginRoutes")); 
app.use("/zones",   require("./routes/zoneRoutes"));  
app.use("/alerts",  require("./routes/alertRoutes")); 
app.use("/team",    require("./routes/teamRoutes"));  
app.use("/workers", require("./routes/workerRoutes"));
app.use("/dashboard", require("./routes/dashboardRoutes"));

// ── WebSocket 서버 (WS_PORT, HTTP와 분리) ──────────────────────────
const wss = new WebSocket.Server({ port: WS_PORT, perMessageDeflate: false });
wss.on("listening", () => console.log(`[WS ] ws://localhost:${WS_PORT}`));
wss.on("connection", (ws, req) => {
  console.log("[WS ] Unity connected:", req?.socket?.remoteAddress);
  ws.on("close",  () => console.log("[WS ] Unity disconnected"));
  ws.on("error",  e  => console.error("[WS ] Error:", e.message));
});

// 공용 브로드캐스트 유틸
function broadcast(type, payload) {
  const msg = JSON.stringify({ type, payload });
  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) client.send(msg);
  });
}

app.get("/", (req, res) => {
  res.json({ ok: true, service: "smart-factory API", now: new Date().toISOString() });
});
app.get("/healthz", (req, res) => res.status(200).send("ok"));

// ── 수집 엔드포인트 (AI→서버→DB 저장→WS 브로드캐스트) ───────────
app.post("/workers", async (req, res) => {
  const data = req.body;
  console.log("[POST] /workers\n", JSON.stringify(data, null, 2));
  try {
    await saveToSQLite(data);         // 콜백형이면 프로미스 래핑 필요
    broadcast("workers_update", data);
    return res.status(200).json({ ok: true });
  } catch (e) {
    console.error("[/workers] DB error:", e);
    return res.status(500).json({ ok: false, error: "DB save failed" });
  }
});

app.post("/zones", async (req, res) => {
  const data = req.body;
  console.log("[POST] /zones\n", JSON.stringify(data, null, 2));
  try {
    await saveToSQLite(data);
    broadcast("zone_update", data);
    return res.status(200).json({ ok: true });
  } catch (e) {
    console.error("[/zones] DB error:", e);
    return res.status(500).json({ ok: false, error: "DB save failed" });
  }
});

app.post("/violations", async (req, res) => {
  const data = req.body;
  console.log("[POST] /violations\n", JSON.stringify(data, null, 2));
  try {
    await saveToSQLite(data);
    broadcast("violation_update", data);
    return res.status(200).json({ ok: true });
  } catch (e) {
    console.error("[/violations] DB error:", e);
    return res.status(500).json({ ok: false, error: "DB save failed" });
  }
});

// ── React 정적 서빙 (배포 시) ───────────────────────────────────────
// app.use(express.static(path.join(__dirname, "frontend", "build")));
// app.get(/^\/(?!api).*/, (req, res) => {
//   res.sendFile(path.join(__dirname, "frontend", "build", "index.html"));
// });

// ── 404 & 에러 핸들러 ───────────────────────────────────────────────
app.use((req, res) => res.status(404).json({ error: "Not Found", path: req.originalUrl }));
app.use((err, req, res, next) => {
  console.error("[ERROR]", err);
  res.status(500).json({ error: "Internal Server Error" });
});

// ── HTTP 서버 시작 ──────────────────────────────────────────────────
app.listen(SERVER_PORT, () => {
  console.log(`[HTTP] http://localhost:${SERVER_PORT}`);
});

module.exports = app;
