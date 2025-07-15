// loginController.js 
const bcrypt = require("bcrypt");
const Admin = require("../models/adminSchema");  // Admin 모델 import

// ✅ 회원가입
const registerAdmin = async (req, res) => {
  const { username, password } = req.body;
  try {
    // 사용자 중복 확인
    const existing = await Admin.findOne({ username });
    if (existing) {
      return res.status(400).json({ message: "이미 존재하는 사용자입니다." });
    }

    // 비밀번호 암호화
    const hashedPassword = await bcrypt.hash(password, 10);

    // 사용자 생성
    const newAdmin = new Admin({
      username,
      password: hashedPassword,
    });

    await newAdmin.save();

    res.status(201).json({ message: "회원가입 성공" });
  } catch (err) {
    console.error("회원가입 에러:", err);
    res.status(500).json({ message: "서버 오류", error: err.message });
  }
};

// ✅ 로그인
const loginUser = async (req, res) => {
  const { username, password } = req.body;
  try {
    const user = await Admin.findOne({ username });
    if (!user) {
      return res.status(400).json({ message: "사용자를 찾을 수 없습니다." });
    }

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(400).json({ message: "비밀번호가 일치하지 않습니다." });
    }

    // 세션, 토큰 등 여기서 처리 가능 (필요 시)
    res.status(200).json({ message: "로그인 성공" });
  } catch (err) {
    console.error("로그인 에러:", err);
    res.status(500).json({ message: "서버 오류", error: err.message });
  }
};

// ✅ 로그아웃 (쿠키/세션 클리어용)
const logout = (req, res) => {
  // 세션이나 쿠키 사용 중이라면 여기서 제거
  res.clearCookie("token");  // JWT 쓴다면 이처럼
  res.status(200).json({ message: "로그아웃 성공" });
};

module.exports = {
  registerAdmin,
  loginUser,
  logout,
};
