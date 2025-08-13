const { getRecentAlerts } = require("../services/alertQueryService");

function handleRecentAlerts(req, res) {
  const limit = parseInt(req.query.limit) || 10;

  getRecentAlerts(limit, (err, result) => {
    if (err) {
      console.error("[RecentAlerts ERROR]", err.message);
      return res.status(500).json({ error: "DB query failed" });
    }

    res.json(result);
  });
}

module.exports = { handleRecentAlerts };
