const asyncHandler = require("express-async-handler");
const bcrypt = require("bcrypt");
const Admin = require("../models/adminSchema"); // ✅ 수정됨
require("dotenv").config();
const jwt = require("jsonwebtoken");
const jwtSecret = process.env.JWT_SECRET;

// GET: 회원가입 페이지
const getRegister = (req, res) => {
    res.send("Register Page");
};

// POST: 회원가입 처리
const registerAdmin = asyncHandler(async (req, res) => {
    const { username, password, password2 } = req.body;
    if (password === password2) {
        const hashPassword = await bcrypt.hash(password, 10);
        const user = await Admin.create({ username, password: hashPassword });
        res.status(201).json({ message: "Register Successful", user });
    } else {
        res.status(400).json({ message: "비밀번호 불일치" });
    }
});

// GET: 로그인 페이지
const getLogin = (req, res) => {
    res.send("Login Page");
};

// POST: 로그인 처리
const loginUser = asyncHandler(async (req, res) => {
    const { username, password } = req.body;
    const user = await Admin.findOne({ username });

    if (!user) {
        return res.status(401).json({ message: "일치하는 사용자 없음" });
    }

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
        return res.status(401).json({ message: "일치하는 비밀번호 없음" });
    }

    const token = jwt.sign({ id: user._id, username: user.username }, jwtSecret, { expiresIn: "1h" });
    res.status(200).json({ message: "로그인 성공", token });
});

// GET: 로그아웃
const logout = (req, res) => {
    res.clearCookie("token");
    res.json({ message: "로그아웃 성공" });
};

module.exports = { getLogin, loginUser, getRegister, registerAdmin, logout };
