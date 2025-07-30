import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

function Register() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [password2, setPassword2] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();

    // 프론트에서 먼저 비밀번호 확인
    if (password !== password2) {
      alert("비밀번호가 일치하지 않습니다.");
      return;
    }


    try {
      const res = await axios.post('http://localhost:5000/register', {
        username,
        password,
        password2, // 이걸 꼭 보내야 백엔드가 비교할 수 있어요
      }, {
        withCredentials: true,  
      });

      alert(res.data.message || '회원가입 성공');
      navigate('/login'); // React 방식 이동
    } catch (err) {
      alert(err.response?.data?.message || '회원가입 실패');
    }
  };

  return (
    <div>
      <h2>회원가입</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="아이디"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
        />
        <input
          type="password"
          placeholder="비밀번호"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        <input
          type="password"
          placeholder="비밀번호 확인"
          value={password2}
          onChange={(e) => setPassword2(e.target.value)}
          required
        />
        <button type="submit">회원가입</button>
      </form>
    </div>
  );
}

export default Register;
