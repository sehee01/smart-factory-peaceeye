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

// 위반 이력 저장소 (메모리 기반)
const ppeViolationHistory = new Map();

// 실시간 워커 위치 데이터 저장소
const workerPositions = new Map();

// 통합된 제한구역 알람 데이터 저장소 (메모리 기반)
const alarmHistory = [];
const activeAlarms = new Map(); // {worker_id_zone_id: alarm_data}

// AI 서버로부터 추론 결과 받기 (기존 /inference 엔드포인트)
app.post("/inference", (req, res) => {
  const data = req.body;
  console.log("[POST] /inference\n", JSON.stringify(data, null, 2));

  // 실시간 워커 위치 업데이트 (항상 업데이트)
  if (data.workers && Array.isArray(data.workers)) {
    data.workers.forEach(worker => {
      workerPositions.set(worker.worker_id, {
        x: worker.x,
        y: worker.y,
        status: worker.status,
        zone_id: worker.zone_id,
        timestamp: worker.timestamp
      });
    });
  }

  // Unity로 실시간 위치 데이터 전송 (항상 전송)
  const realtimeData = {
    type: "worker_positions",
    workers: Array.from(workerPositions.values()),
    timestamp: new Date().toISOString()
  };
  wss.clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify(realtimeData));
    }
  });

  // 위반 데이터 처리 (위반이 있는 경우에만)
  let violationData = null;

  // PPE 위반 이력 업데이트
  if (data.ppe_violations && Array.isArray(data.ppe_violations) && data.ppe_violations.length > 0) {
    console.log("[PPE 위반 감지] 위반 수:", data.ppe_violations.length);
    
    data.ppe_violations.forEach(violation => {
      const key = violation.worker_id;
      if (!ppeViolationHistory.has(key)) {
        ppeViolationHistory.set(key, []);
      }
      
      // 위반 타입이 배열인 경우 처리
      if (Array.isArray(violation.violation_type)) {
        console.log(`[PPE 위반] ${violation.worker_id}: ${violation.violation_type.join(', ')} (${violation.violation_count} violations)`);
      } else {
        console.log(`[PPE 위반] ${violation.worker_id}: ${violation.violation_type}`);
      }
      
      ppeViolationHistory.get(key).push(violation);
      
      // 최근 10개만 유지
      if (ppeViolationHistory.get(key).length > 10) {
        ppeViolationHistory.get(key).shift();
      }
    });
    
    // PPE 위반 데이터 준비
    violationData = {
      type: "ppe_violations",
      violations: data.ppe_violations,
      violation_history: Object.fromEntries(ppeViolationHistory),
      timestamp: new Date().toISOString()
    };
  }

  // 제한구역 위반은 이제 /alerts 엔드포인트에서 처리됨 (통합된 알람 시스템)
  // 기존 restricted_violations 데이터는 무시하고 알람 시스템 사용

  // 위반이 있는 경우에만 Unity로 위반 데이터 전송
  if (violationData) {
    wss.clients.forEach(client => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(JSON.stringify(violationData));
      }
    });
  }

  res.sendStatus(200);
});

// workers 위치/상태 정보 (기존 app.js 형식)
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

// zone 통계 정보 (기존 app.js 형식)
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

// violations (ppe, roi) 정보 (기존 app.js 형식)
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

// 알람 데이터 처리 (ROI 위험구역 알람)
app.post("/alerts", (req, res) => {
  try {
    const { alerts } = req.body;
    console.log("[POST] /alerts\n", JSON.stringify(req.body, null, 2));
    
    if (!alerts || !Array.isArray(alerts)) {
      return res.status(400).json({ error: "Invalid alerts data format" });
    }
    
    // 알람 데이터를 SQLite에 저장
    alerts.forEach(alert => {
      const alarmData = {
        worker_id: alert.worker_id,
        zone_id: alert.zone_id,
        alert_type: alert.type,
        message: alert.message,
        distance: alert.distance,
        timestamp: alert.timestamp,
        requires_action: alert.requires_immediate_action,
        severity: alert.severity,
        color: alert.color,
        zone_type: alert.zone_type
      };
      
      saveToSQLite({ alerts: [alarmData] });
      
      // 알람 히스토리에 추가
      alarmHistory.push(alarmData);
      
      // 최근 1000개 알람만 유지
      if (alarmHistory.length > 1000) {
        alarmHistory.shift();
      }
      
      // 활성 알람 관리
      const alarmKey = `${alert.worker_id}_${alert.zone_id}`;
      if (alert.type === 'clear') {
        // 알람 해제
        activeAlarms.delete(alarmKey);
        console.log(`[ALARM CLEAR] ${alert.worker_id} in ${alert.zone_id}`);
      } else {
        // 알람 활성화
        activeAlarms.set(alarmKey, alarmData);
        console.log(`[ALARM ${alert.type.toUpperCase()}] ${alert.worker_id} in ${alert.zone_id} (${alert.distance?.toFixed(1)}m)`);
      }
    });
    
    // WebSocket으로 실시간 알람 브로드캐스트
    wss.clients.forEach(client => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(JSON.stringify({
          type: "alert_update",
          data: alerts
        }));
      }
    });
    
    res.status(200).json({ 
      message: "Alerts processed successfully", 
      count: alerts.length,
      active_alarms: activeAlarms.size
    });
    
  } catch (error) {
    console.error("Alert processing error:", error);
    res.status(500).json({ error: "Failed to process alerts" });
  }
});

// 위반 이력 조회 API
app.get("/api/violation-history/:worker_id", (req, res) => {
  const workerId = req.params.worker_id;
  
  // 해당 워커의 제한구역 알람 필터링
  const workerAlarms = alarmHistory.filter(alarm => 
    alarm.worker_id === workerId && alarm.alert_type !== 'clear' && alarm.zone_type === 'restricted'
  );
  
  res.json({
    ppe_violations: ppeViolationHistory.get(workerId) || [],
    restricted_violations: workerAlarms // 제한구역 위반을 알람 데이터로 대체
  });
});

// 모든 위반 이력 조회 API
app.get("/api/violation-history", (req, res) => {
  // 제한구역 위반을 알람 데이터로 그룹화
  const restrictedViolationsByWorker = {};
  alarmHistory.forEach(alarm => {
    if (alarm.alert_type !== 'clear' && alarm.zone_type === 'restricted') {
      if (!restrictedViolationsByWorker[alarm.worker_id]) {
        restrictedViolationsByWorker[alarm.worker_id] = [];
      }
      restrictedViolationsByWorker[alarm.worker_id].push(alarm);
    }
  });
  
  res.json({
    ppe_violations: Object.fromEntries(ppeViolationHistory),
    restricted_violations: restrictedViolationsByWorker
  });
});

// 위반 이력 초기화 API
app.delete("/api/violation-history", (req, res) => {
  ppeViolationHistory.clear();
  // 알람 히스토리는 별도 엔드포인트에서 관리
  res.json({ message: "위반 이력이 초기화되었습니다." });
});

// 알람 히스토리 조회 API (제한구역 위반 포함)
app.get("/alerts", (req, res) => {
  try {
    const { worker_id, zone_id, limit = 100, type, zone_type } = req.query;
    
    let filteredAlarms = [...alarmHistory];
    
    // 필터링
    if (worker_id) {
      filteredAlarms = filteredAlarms.filter(alarm => alarm.worker_id === worker_id);
    }
    
    if (zone_id) {
      filteredAlarms = filteredAlarms.filter(alarm => alarm.zone_id === zone_id);
    }
    
    if (type) {
      filteredAlarms = filteredAlarms.filter(alarm => alarm.alert_type === type);
    }
    
    if (zone_type) {
      filteredAlarms = filteredAlarms.filter(alarm => alarm.zone_type === zone_type);
    }
    
    // 최신순 정렬 및 제한
    filteredAlarms.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    filteredAlarms = filteredAlarms.slice(0, parseInt(limit));
    
    res.json({
      alerts: filteredAlarms,
      total_count: filteredAlarms.length,
      active_alarms: Array.from(activeAlarms.values()),
      summary: {
        critical_alarms: filteredAlarms.filter(a => a.severity === 'critical').length,
        warning_alarms: filteredAlarms.filter(a => a.severity === 'warning').length,
        cleared_alarms: filteredAlarms.filter(a => a.alert_type === 'clear').length
      }
    });
    
  } catch (error) {
    console.error("Failed to fetch alerts:", error);
    res.status(500).json({ error: "Failed to fetch alerts" });
  }
});

// 활성 알람 조회 API
app.get("/alerts/active", (req, res) => {
  try {
    const activeAlarmsList = Array.from(activeAlarms.values());
    res.json({
      active_alarms: activeAlarmsList,
      count: activeAlarmsList.length
    });
  } catch (error) {
    console.error("Failed to fetch active alerts:", error);
    res.status(500).json({ error: "Failed to fetch active alerts" });
  }
});

// 알람 통계 API
app.get("/alerts/stats", (req, res) => {
  try {
    const now = new Date();
    const oneHourAgo = new Date(now.getTime() - 60 * 60 * 1000);
    const oneDayAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000);
    
    const hourlyAlarms = alarmHistory.filter(alarm => 
      new Date(alarm.timestamp) > oneHourAgo
    );
    
    const dailyAlarms = alarmHistory.filter(alarm => 
      new Date(alarm.timestamp) > oneDayAgo
    );
    
    const stats = {
      total_alarms: alarmHistory.length,
      active_alarms: activeAlarms.size,
      hourly_alarms: hourlyAlarms.length,
      daily_alarms: dailyAlarms.length,
      critical_alarms: alarmHistory.filter(a => a.severity === 'critical').length,
      warning_alarms: alarmHistory.filter(a => a.severity === 'warning').length,
      cleared_alarms: alarmHistory.filter(a => a.alert_type === 'clear').length
    };
    
    res.json(stats);
  } catch (error) {
    console.error("Failed to fetch alert stats:", error);
    res.status(500).json({ error: "Failed to fetch alert stats" });
  }
});

// 알람 히스토리 초기화 API
app.delete("/alerts", (req, res) => {
  try {
    alarmHistory.length = 0;
    activeAlarms.clear();
    res.json({ message: "알람 히스토리가 초기화되었습니다." });
  } catch (error) {
    console.error("Failed to clear alerts:", error);
    res.status(500).json({ error: "Failed to clear alerts" });
  }
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