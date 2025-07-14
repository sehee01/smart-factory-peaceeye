const express = require("express");
const router = express.Router();
const {
  getLogin,
  loginUser,
  getRegister,
  registerAdmin,
  logout,
} = require("../controller/loginController");

router.route("/").get(getLogin).post(loginUser);

router.route("/register").get(getRegister).post(registerAdmin);

router.route("/logout").get(logout);

// ✅ 이거 꼭 필요!
module.exports = router;
