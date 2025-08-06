const { getRealtimeZones, getZoneHistory } = require("../services/zoneQueryService");

function handleRealtimeZones(req, res) {
  getRealtimeZones((err, result) => {
    if (err) {
      console.error("[RealtimeZone ERROR]", err.message);
      return res.status(500).json({ error: "DB query failed" });
    }
    res.json(result);
  });
}

function handleZoneHistory(req, res) {
  const zoneId = req.params.id;

  getZoneHistory(zoneId, (err, result) => {
    if (err) {
      console.error("[ZoneHistory ERROR]", err.message);
      return res.status(500).json({ error: "DB query failed" });
    }

    res.json({
      zoneId,
      history: result
    });
  });
}

module.exports = { handleRealtimeZones, handleZoneHistory };
