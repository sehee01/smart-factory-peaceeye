import React, { useState } from "react";
import axios from "axios";
import { useNavigate, Link } from "react-router-dom";
import "./Login.css";

function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [loading, setLoading] = useState(false);
  const [errMsg, setErrMsg] = useState("");
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrMsg("");
    setLoading(true);
    try {
      await axios.post(
        process.env.REACT_APP_API_BASE ?? "http://localhost:5000/login",
        { username, password },
        { withCredentials: true }
      );
      navigate("/dashboard");
    } catch (err) {
      setErrMsg(err.response?.data?.message || "로그인에 실패했습니다.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-page">
      {/* 좌측 히어로 영역 */}
      <div className="hero">
        <div className="brand">
          <div className="logo-dot" />
          <span className="brand-top">INDUSTRIAL SAFETY</span>
          <span className="brand-bottom">MONITORING</span>
        </div>

        {/* 데코: 점선 구역 + 아바타만 유지 */}
        <div className="zone">
          <div className="zone-dash" />
          <div className="avatar avatar--green" />
          <div className="avatar avatar--red" />
        </div>

        <div className="footer-note">
          Real-time worker status • PPE detection • Risk zones
        </div>
      </div>

      {/* 우측 로그인 카드 */}
      <div className="panel">
        <div className="panel-inner">
          <h1 className="title">Welcome back</h1>
          <p className="subtitle">로그인하여 대시보드에 접속하세요.</p>

          {errMsg && <div className="alert">{errMsg}</div>}

          <form onSubmit={handleSubmit} className="form">
            <label className="label">
              아이디
              <input
                className="input"
                type="text"
                placeholder="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                autoComplete="username"
                required
              />
            </label>

            <label className="label">
              비밀번호
              <div className="pw-wrap">
                <input
                  className="input"
                  type={showPw ? "text" : "password"}
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="current-password"
                  required
                />
                <button
                  type="button"
                  className="pw-toggle"
                  onClick={() => setShowPw((v) => !v)}
                  aria-label="비밀번호 표시 전환"
                >
                  {showPw ? "Hide" : "Show"}
                </button>
              </div>
            </label>

            <div className="form-row">
              <label className="check">
                <input type="checkbox" /> 로그인 상태 유지
              </label>
              <Link to="/forgot" className="link">
                비밀번호 찾기
              </Link>
            </div>

            <button className="btn" type="submit" disabled={loading}>
              {loading ? "로그인 중..." : "로그인"}
            </button>
          </form>

          <div className="divider">
            <span>또는</span>
          </div>

          <div className="social-row">
            <button className="btn btn--ghost" type="button" disabled>
              Google
            </button>
            <button className="btn btn--ghost" type="button" disabled>
              GitHub
            </button>
          </div>

          <p className="signup">
            계정이 없으신가요?{" "}
            <Link to="/register" className="link-strong">
              회원가입
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}

export default Login;
