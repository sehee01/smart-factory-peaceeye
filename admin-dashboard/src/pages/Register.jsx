import React, { useState } from 'react';
import axios from 'axios';

function Register() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [password2, setPassword2] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post('/api/register', { username, password, password2 });
      alert('회원가입 성공!');
      window.location.href = '/';
    } catch (err) {
      alert(err.response.data.message || '회원가입 실패');
    }
  };

  return (
    <div>
      <h2>회원가입</h2>
      <form onSubmit={handleSubmit}>
        <input type="text" placeholder="아이디" value={username} onChange={e => setUsername(e.target.value)} required />
        <input type="password" placeholder="비밀번호" value={password} onChange={e => setPassword(e.target.value)} required />
        <input type="password" placeholder="비밀번호 확인" value={password2} onChange={e => setPassword2(e.target.value)} required />
        <button type="submit">회원가입</button>
      </form>
    </div>
  );
}

export default Register;
