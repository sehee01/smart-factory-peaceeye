const db = require("../config/dbConnect");

function getTeamWorkers(teamId, callback) {
  const query = `
    SELECT 
      w.worker_id,
      w.name,
      w.position,
      COALESCE(SUM(z.product_count), 0) AS total_products,
      COALESCE((
        SELECT COUNT(*) 
        FROM worker_alerts a 
        WHERE a.worker_id = w.worker_id
      ), 0) AS total_alerts
    FROM workers w
    LEFT JOIN zone_realtime_data z ON w.worker_id = z.worker_id
    WHERE w.team_id = ?
    GROUP BY w.worker_id
  `;

  db.all(query, [teamId], (err, rows) => {
    if (err) return callback(err);

    const formatted = rows.map(row => ({
      workerId: row.worker_id,
      name: row.name,
      position: row.position,
      totalProducts: Number(row.total_products),
      totalAlerts: Number(row.total_alerts)
    }));

    callback(null, formatted);
  });
}

module.exports = { getTeamWorkers };
