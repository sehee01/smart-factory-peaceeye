const bcrypt = require("bcrypt");
const db = require("../config/dbConnect");  // SQLite DB 연결
const jwt = require("jsonwebtoken");

// 환경변수
const JWT_SECRET = process.env.JWT_SECRET;
const JWT_EXPIRES_IN = process.env.JWT_EXPIRES_IN || '1d';

// 회원가입
const registerAdmin = async (req, res) => {
  const { username, password } = req.body;
  try {
    // 사용자 중복 확인
    const existing = await db("admins").where({ username }).first();
    if (existing) {
      return res.status(400).json({ message: "이미 존재하는 사용자입니다." });
    }

    // 비밀번호 암호화
    const hashedPassword = await bcrypt.hash(password, 10);

    // 사용자 저장
    await db("admins").insert({
      username,
      password: hashedPassword,
    });

    res.status(201).json({ message: "회원가입 성공" });
  } catch (err) {
    console.error("회원가입 에러:", err);
    res.status(500).json({ message: "서버 오류", error: err.message });
  }
};

// 로그인 + JWT 토큰 생성 + 쿠키 저장
const loginUser = async (req, res) => {
  const { username, password } = req.body;
  try {
    const user = await db("admins").where({ username }).first();
    if (!user) {
      return res.status(400).json({ message: "사용자를 찾을 수 없습니다." });
    }

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(400).json({ message: "비밀번호가 일치하지 않습니다." });
    }

    // JWT 토큰 생성
    const token = jwt.sign(
      { id: user.id, username: user.username },
      JWT_SECRET,
      { expiresIn: JWT_EXPIRES_IN }
    );

    // 토큰을 httpOnly 쿠키로 저장
    res.cookie("token", token, {
      httpOnly: true,
      secure: false, // 배포 시 true + HTTPS 필요
      maxAge: 24 * 60 * 60 * 1000, // 1일
    });

    res.status(200).json({ message: "로그인 성공", username: user.username });
  } catch (err) {
    console.error("로그인 에러:", err);
    res.status(500).json({ message: "서버 오류", error: err.message });
  }
};

// 로그아웃 (쿠키/세션 클리어용)
const logout = (req, res) => {
  res.clearCookie("token");
  res.status(200).json({ message: "로그아웃 성공" });
};

module.exports = {
  registerAdmin,
  loginUser,
  logout,
};
