const express = require("express");
const router = express.Router();
const { handleRealtimeZones, handleZoneHistory } = require("../controllers/zoneController");

// 실시간
router.get("/realtime", handleRealtimeZones); 
// GET /zones/:id/history
router.get("/:id/history", handleZoneHistory);

module.exports = router;
