const jwt = require("jsonwebtoken");

exports.verifyToken = (req, res, next) => {
    const authHeader = req.headers.authorization;

    if (!authHeader || !authHeader.startsWith("Bearer ")) {
        return res.status(401).json({ message: "토큰이 없습니다." });
    }

    const token = authHeader.split(" ")[1];

    try {
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        req.user = decoded;  // 이후 req.user로 접근 가능
        next();
    } catch (err) {
        return res.status(403).json({ message: "토큰이 유효하지 않습니다." });
    }
};
