import React, { useEffect, useRef } from 'react';

export default function FactoryMap() {
  const canvasRef = useRef(null);

  useEffect(() => {
    /* ===== Unity 인스턴스 로드 ===== */
    const buildUrl   = "/unity/Build";
    const loaderUrl  = buildUrl + "/Build.loader.js";     // ← ★ 빌드 이름
    const config = {
      dataUrl:        buildUrl + "/Build.data",
      frameworkUrl:   buildUrl + "/Build.framework.js",
      codeUrl:        buildUrl + "/Build.wasm",
      streamingAssetsUrl: "StreamingAssets",
      companyName:    "PeaceEye",
      productName:    "FactoryMonitor",
      productVersion: "1.0",
    };

    const script = document.createElement("script");
    script.src = loaderUrl;
    script.onload = () => {
      window
        .createUnityInstance(canvasRef.current, config, (progress) => {
          console.log("Unity loading...", progress);
        })
        .then((unityInstance) => {
          window.UnityInstance = unityInstance;        // ← 다른 컴포넌트에서도 접근 가능
          console.log("✅ Unity ready");
        })
        .catch((msg) => console.error("Unity load error:", msg));
    };
    document.body.appendChild(script);

    /* === cleanup === */
    return () => {
      script.remove();
      window.UnityInstance.Quit();
    };
  }, []);

  return (
    <div style={{ width: '100%', height: 400, background: '#000' }}>
      <canvas
        id="unity-canvas" 
        ref={canvasRef}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
}
