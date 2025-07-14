import React, { useEffect, useRef } from 'react';

export default function FactoryMap() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const buildUrl = "/unity/Build";
    const config = {
      dataUrl: buildUrl + "/Build.data",
      frameworkUrl: buildUrl + "/Build.framework.js",
      codeUrl: buildUrl + "/Build.wasm",
      streamingAssetsUrl: "StreamingAssets",
      companyName: "PeaceEye",
      productName: "FactoryMonitor",
      productVersion: "1.0",
    };

    const script = document.createElement("script");
    script.src = config.frameworkUrl.replace("framework.js", "loader.js");
    script.onload = () => {
      window
        .createUnityInstance(canvasRef.current, config, (progress) => {
          console.log("Unity loading...", progress);
        })
        .then((unityInstance) => {
          window.UnityInstance = unityInstance;
          console.log("✅ Unity ready");
        })
        .catch((msg) => console.error("Unity load error:", msg));
    };
    document.body.appendChild(script);

    return () => {
      script.remove();
      if (window.UnityInstance) {
        window.UnityInstance.Quit();
      }
    };
  }, []);

  return (
    <div style={{ width: '100%', height: '100vh', background: '#000' }}>
      <canvas ref={canvasRef} id="unity-canvas" style={{ width: '100%', height: '100%' }} />
    </div>
  );
}
