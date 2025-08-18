import React, { useState, useEffect, useCallback } from 'react';
import { Unity, useUnityContext } from 'react-unity-webgl';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

// Unity 빌드 파일 경로 (public 폴더 기준)
const unityBuildFolder = "/Build";
const unityBuildJson = "Build.loader.js"; 

function Dashboard() {
  const [user, setUser] = useState(null);
  const [isAuthenticating, setIsAuthenticating] = useState(true);
  const navigate = useNavigate();

  // react-unity-webgl 설정
  const { unityProvider, isLoaded, loadingProgression } = useUnityContext({
    // Unity 빌드 파일 경로 참고할 것
    loaderUrl: `${unityBuildFolder}/${unityBuildJson}`,
    dataUrl: "/unity/Build/Build.data.unityweb",
    frameworkUrl: "/unity/Build/Build.framework.js.unityweb",
    codeUrl: "/unity/Build/Build.wasm.unityweb",
  });

  // 로그인 상태 확인 (백엔드 쿠키 기반 JWT 인증)
  useEffect(() => {
    axios.get('http://localhost:5000/me', {
      withCredentials: true
    }).then(res => {
      setUser(res.data.user);
      console.log("로그인 사용자:", res.data.user.username);
    }).catch(err => {
      console.warn("인증 실패 또는 로그인 만료:", err.response?.data?.message);
      navigate('/login'); // 로그인 페이지로 이동 ('/' 혹은 '/login')
    }).finally(() => {
      setIsAuthenticating(false); // 인증 절차 완료
    });
  }, [navigate]);

  // Unity로부터 로그아웃 신호를 처리하는 함수
  const handleLogoutFromUnity = useCallback(async () => {
    console.log("Unity로부터 로그아웃 신호를 받았습니다.");
    try {
      await axios.post("http://localhost:5000/logout", {}, {
        withCredentials: true
      });
      alert("성공적으로 로그아웃되었습니다.");
      navigate("/"); // 로그아웃 성공 시, 초기 페이지(로그인)로 이동
    } catch (error) {
      console.error("로그아웃 처리 중 오류 발생:", error);
      alert("로그아웃에 실패했습니다. 관리자에게 문의하세요.");
      navigate("/");
    }
  }, [navigate]);

  // Unity가 호출할 함수를 전역(window)에 등록
  useEffect(() => {
    window.handleUnityLogout = handleLogoutFromUnity;
    return () => {
      delete window.handleUnityLogout;
    };
  }, [handleLogoutFromUnity]); 

  // 인증 확인 중일 때 로딩 메시지 표시
  if (isAuthenticating) {
    return <div style={{ textAlign: 'center', paddingTop: '20%' }}>인증 정보를 확인 중입니다...</div>;
  }
  // 인증 실패로 user가 없을 경우 렌더링하지 않음 (useEffect에서 이미 페이지 이동 처리)
  if (!user) return null;

  // 5. 최종 렌더링
  return (
    <div className="unity-container" style={{ width: '100vw', height: '100vh', margin: 0, padding: 0, overflow: 'hidden' }}>
      {/* Unity 로딩 오버레이 */}
      {!isLoaded && (
        <div className="loading-overlay" style={{ position: 'absolute', width: '100%', height: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center', background: 'rgba(0,0,0,0.7)', color: 'white' }}>
          <p>게임 로딩 중... {Math.round(loadingProgression * 100)}%</p>
        </div>
      )}
      {/* Unity 게임 캔버스 */}
      <Unity
        unityProvider={unityProvider}
        style={{
          width: '100%',
          height: '100%',
          display: 'block',
          background: '#000', // 옵션: 배경이 안 보이면 검정색
          visibility: isLoaded ? "visible" : "hidden"
        }}
      />
    </div>
  );
}

export default Dashboard;
