const db = require("../config/dbConnect");

function getRecentAlerts(limit, callback) {
  const query = `
    SELECT worker_id, zone_id, type, timestamp
    FROM worker_alerts
    ORDER BY timestamp DESC
    LIMIT ?
  `;

  db.all(query, [limit], (err, rows) => {
    if (err) return callback(err);

    const formatted = rows.map(row => ({
      workerId: row.worker_id,
      zoneId: row.zone_id,
      type: row.type,
      timestamp: row.timestamp
    }));

    callback(null, formatted);
  });
}

module.exports = { getRecentAlerts };
