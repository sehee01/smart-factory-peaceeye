const express = require("express");
const path = require("path");
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

// SQLite 초기화 추가
require("./config/initDB"); // users 테이블 초기화

// 로그인/회원가입 라우터 연결 (SQLite 기반 컨트롤러를 내부에서 사용)
app.use("/", require("./routes/loginRoutes"));

// React 빌드 결과물 정적 파일로 서빙
app.use(express.static(path.join(__dirname, "frontend/build")));  // client 폴더에 React 있는 경우

// 모든 GET 요청은 React index.html 반환 (정규식 사용)
app.get(/^\/(?!api).*/, (req, res) => {
  res.sendFile(path.join(__dirname, "frontend", "build", "index.html"));
});

// 서버 시작
const PORT = process.env.PORT;
app.listen(PORT, () => {
  console.log(`${PORT}번 포트에서 서버 실행 중`);
});

