const express = require("express");
const router = express.Router();
const { handleTeamWorkers } = require("../controllers/teamController");

// GET /team/:team_id/workers
router.get("/:team_id/workers", handleTeamWorkers);

module.exports = router;
