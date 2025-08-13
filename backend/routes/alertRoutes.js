const express = require("express");
const router = express.Router();
const { handleRecentAlerts } = require("../controllers/alertController");

// GET /alerts/recent
router.get("/recent", handleRecentAlerts);

module.exports = router;
