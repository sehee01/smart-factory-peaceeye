// routes/loginRoutes.js
const express = require('express');
const router = express.Router();
const { registerAdmin, loginUser, logout } = require('../controllers/loginController');
const verifyToken = require('../middlewares/authMiddleware');  // 인증 미들웨어 import

// 회원가입
router.post('/register', registerAdmin);

// 로그인
router.post('/login', loginUser);

// 로그아웃
router.post('/logout', logout);

// 로그인 유지 확인 (인증 필요)
router.get('/me', verifyToken, (req, res) => {
  res.status(200).json({ message: "인증된 사용자입니다", user: req.user });
});

module.exports = router;