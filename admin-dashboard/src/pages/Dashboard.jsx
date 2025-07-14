import React, { useState, useEffect } from 'react';
import { Container, Grid } from '@mui/material';
import FactoryMap from '../components/FactoryMap';

// 🧪 임시 좌표 계산 함수 (worker ID에 따라 랜덤 위치 생성)
const workerIdToXY = (workerId) => {
  const seed = parseInt(workerId.toString().slice(-2)) || 1;
  const x = (seed * 73) % 500;
  const y = (seed * 91) % 300;
  return { x, y };
};

// API URL은 환경변수에서 가져옴
const apiBase = process.env.REACT_APP_API_URL;

export default function Dashboard() {
  const [rows, setRows] = useState([]);

  // 🔄 최초 렌더링 시 백엔드에서 로그 목록 가져오기
  useEffect(() => {
    fetch(`${apiBase}/log`)
      .then(res => res.json())
      .then(data => {
        console.log('📋 기존 로그:', data);
        setRows(data.slice(-20).reverse().map((item, index) => ({
          id: index + 1,
          worker: item.worker_id,
          event: item.event_type,
          time: item.timestamp || new Date().toLocaleTimeString()
        })));
      })
      .catch(err => {
        console.error('❌ 로그 불러오기 실패', err);

        // 🔁 실패 시 임시 테스트용 데이터
        const dummy = [
          { worker_id: "W-001", event_type: "Entered danger zone" },
          { worker_id: "W-002", event_type: "Exited danger zone" }
        ];
        setRows(dummy.map((d, i) => ({
          id: i + 1,
          worker: d.worker_id,
          event: d.event_type,
          time: new Date().toLocaleTimeString()
        })));

        const pos = workerIdToXY(dummy[0].worker_id);
        if (window.UnityInstance) {
          window.UnityInstance.SendMessage(
            "WorkerController",
            "SetWorkerPosition",
            JSON.stringify({
              id: dummy[0].worker_id,
              x: pos.x,
              y: pos.y
            })
          );
        }
      });
  }, []);

  return (
    <Container maxWidth="lg" sx={{ mt: 10 }}>
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <FactoryMap />
        </Grid>
        <Grid item xs={12} md={4}>
          {/* 추가 UI 영역 */}
        </Grid>
        <Grid item xs={12}>
        </Grid>
      </Grid>

    </Container>
  );
}
