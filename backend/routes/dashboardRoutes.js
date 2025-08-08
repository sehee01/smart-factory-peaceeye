const express = require("express");
const router = express.Router();
const {
  getOngoingTasks,
  getPartSummary,
  getWorkerStatus,
  getRiskSummary,
  exportCsv
} = require("../controllers/dashboardController");

// 최소 DTO
router.get("/ongoing-tasks", getOngoingTasks);
router.get("/part-summary", getPartSummary);
router.get("/worker-status", getWorkerStatus);
router.get("/risk-summary", getRiskSummary);
router.get("/export-csv", exportCsv);

module.exports = router;
