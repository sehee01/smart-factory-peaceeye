import React, { useState } from 'react';
import axios from 'axios';

function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post('/api/login', { username, password }, {
        withCredentials: true  // 쿠키 저장 허용
      });
      alert('로그인 성공');
      window.location.href = '/dashboard'; // 성공 시 이동
    } catch (err) {
      alert(err.response.data.message || '로그인 실패');
    }
  };

  return (
    <div>
      <h2>로그인</h2>
      <form onSubmit={handleSubmit}>
        <input type="text" placeholder="아이디" value={username} onChange={e => setUsername(e.target.value)} required />
        <input type="password" placeholder="비밀번호" value={password} onChange={e => setPassword(e.target.value)} required />
        <button type="submit">로그인</button>
      </form>
    </div>
  );
}

export default Login;
