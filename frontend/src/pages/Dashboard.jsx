import React from 'react';
import { Unity, useUnityContext } from 'react-unity-webgl';

function Dashboard() {
  const { unityProvider } = useUnityContext({
    loaderUrl: "/unity/Build/Build.loader.js",
    dataUrl: "/unity/Build/Build.data",
    frameworkUrl: "/unity/Build/Build.framework.js",
    codeUrl: "/unity/Build/Build.wasm",
  });

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
