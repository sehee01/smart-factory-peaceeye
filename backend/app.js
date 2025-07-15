// backend/app.js
const express = require("express");
const app = express();
const cookieParser = require("cookie-parser");
const cors = require("cors");
const dbConnect = require("./config/dbConnect");
require("dotenv").config();

// 미들웨어
app.use(cors({
  origin: "http://localhost:3000",  // React 앱 주소
  credentials: true                 // 쿠키 포함 허용
}));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(cookieParser());

// DB 연결
dbConnect();

// ✅ 여기 중요! 모든 로그인/회원가입 요청은 /api로 시작되도록 함
app.use("/api", require("./routes/loginRoutes"));

// 서버 시작
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`${PORT}번 포트에서 서버 실행 중`);
});

