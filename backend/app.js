// 1. 모듈 불러오기
const express = require("express");
const dotenv = require("dotenv");
const cors = require("cors");

dotenv.config();

// 2. express 앱 생성
const app = express();

// 3. 미들웨어 설정
app.use(cors());
app.use(express.json()); // JSON 파싱

// 4. 라우터 등록 (항상 app 객체 생성 이후에!)
app.use("/api", require("./routes/loginRoutes"));        // 로그인, 회원가입
app.use("/api", require("./routes/protectedRoutes"));    // 보호된 API

// 5. 서버 실행
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`🚀 Server is running on port ${PORT}`);
});
