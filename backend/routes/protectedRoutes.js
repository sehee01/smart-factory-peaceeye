const express = require("express");
const router = express.Router();
const { verifyToken } = require("../middleware/authMiddleware");

router.get("/updates", verifyToken, (req, res) => {
    res.json({
        message: "안전하게 보호된 API입니다.",
        user: req.user
    });
});

module.exports = router;
