const express = require("express");
const router = express.Router();
const { getWorkerHistory } = require("../controllers/workerController");

// GET /workers/:worker_id/history
router.get("/:worker_id/history", getWorkerHistory);

module.exports = router;
