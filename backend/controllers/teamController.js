const { getTeamWorkers } = require("../services/teamQueryService");

function handleTeamWorkers(req, res) {
  const teamId = req.params.team_id;

  getTeamWorkers(teamId, (err, result) => {
    if (err) {
      console.error("[TeamWorkers ERROR]", err.message);
      return res.status(500).json({ error: "DB query failed" });
    }

    res.json(result);
  });
}

module.exports = { handleTeamWorkers };
