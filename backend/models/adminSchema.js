const mongoose = require("mongoose");

const adminSchema = new mongoose.Schema({
  username: {
    type: String,
    required: true,
    unique: true, // 중복 방지
    trim: true
  },
  password: {
    type: String,
    required: true
  }
}, {
  timestamps: true // 생성일, 수정일 자동 기록
});

module.exports = mongoose.model("Admin", adminSchema);
