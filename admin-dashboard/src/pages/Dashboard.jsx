import React, { useState, useEffect } from 'react';
import { Container, Grid } from '@mui/material';
import FactoryMap from '../components/FactoryMap';
import AlertBox from '../components/AlertBox';
import LogTable from '../components/LogTable';
import { io } from 'socket.io-client';

// 소켓 서버 주소
const socket = io('http://localhost:3001'); // 백엔드 포트에 따라 조정

export default function Dashboard() {
  const [alert, setAlert] = useState({ open: false, msg: '' });
  const [rows, setRows] = useState([]);

  useEffect(() => {
    socket.on('connect', () => {
      console.log('✅ WebSocket connected');
    });

    socket.on('workerEvent', (data) => {
      console.log('📩 수신한 이벤트:', data);

      const newRow = {
        id: rows.length + 1,
        worker: data.worker_id,
        event: data.event_type,
        time: new Date().toLocaleTimeString()
      };

      setRows(prev => [newRow, ...prev.slice(0, 19)]); // 최대 20개 유지
      setAlert({ open: true, msg: `${data.worker_id} - ${data.event_type}` });
      
      /*  === Unity로 좌표 전달 === */
      const pos = workerIdToXY(data.worker_id);   // ↖︎ (예) id를 xy로 변환하는 함수
      if (window.UnityInstance) {
        window.UnityInstance.SendMessage(
          "WorkerController",          // Unity C# 스크립트 붙은 GameObject 이름
          "SetWorkerPosition",         // C# public 함수 이름
          JSON.stringify({             // 전달할 데이터
            id: data.worker_id,
            x: pos.x,
            y: pos.y
          })
        );
      }
    });

    return () => {
      socket.off('workerEvent');
    };
  }, [rows]);

  return (
    <Container maxWidth="lg" sx={{ mt: 4 }}>
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <FactoryMap />
        </Grid>
        <Grid item xs={12} md={4}>
          {/* 추가 UI 영역 */}
        </Grid>
        <Grid item xs={12}>
          <LogTable rows={rows} />
        </Grid>
      </Grid>

      <AlertBox
        open={alert.open}
        message={alert.msg}
        onClose={() => setAlert({ ...alert, open: false })}
      />
    </Container>
  );
}