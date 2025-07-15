// routes/loginRoutes.js
const express = require('express');
const router = express.Router();
const { registerAdmin, loginUser, logout } = require('../controllers/loginController');

// ❌ 페이지 보여주는 GET 라우트는 제거 (React가 하니까 필요 없음)

// ✅ 회원가입 처리
router.post('/register', registerAdmin);

// ✅ 로그인 처리
router.post('/login', loginUser);

// ✅ 로그아웃 처리
router.post('/logout', logout);

module.exports = router;
