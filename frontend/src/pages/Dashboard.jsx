import React, { useState, useEffect } from 'react';
import { Unity, useUnityContext } from 'react-unity-webgl';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

function Dashboard() {
  const [user, setUser] = useState(null);
  const navigate = useNavigate();

  // 로그인 상태 확인 (백엔드 쿠키 기반 JWT 인증)
  useEffect(() => {
    axios.get('http://localhost:5000/me', {
      withCredentials: true
    }).then(res => {
      setUser(res.data.user); // 예: { id: 1, username: 'admin' }
      console.log("로그인 사용자:", res.data.user.username);
    }).catch(err => {
      console.warn("인증 실패 또는 로그인 만료:", err.response?.data?.message);
      navigate('/login'); // 로그인 페이지로 이동
    });
  }, [navigate]);

  const { unityProvider } = useUnityContext({
    loaderUrl: "/unity/Build/Build.loader.js",
    dataUrl: "/unity/Build/Build.data",
    frameworkUrl: "/unity/Build/Build.framework.js",
    codeUrl: "/unity/Build/Build.wasm",
  });

  // 인증 확인되기 전에는 Unity 표시 X
  if (!user) return null;

  return (
    <div style={{ width: '100vw', height: '100vh', margin: 0, padding: 0, overflow: 'hidden' }}>
      <Unity
        unityProvider={unityProvider}
        style={{
          width: '100%',
          height: '100%',
          display: 'block',
          background: '#000' // 옵션: 배경이 안 보이면 검정색
        }}
      />
    </div>
  );
}

export default Dashboard;
