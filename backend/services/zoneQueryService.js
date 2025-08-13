const db = require("../config/dbConnect"); 

// 실시간 조회: 각 zone의 최신 1개
function getRealtimeZones(callback) {
  db.raw(`
    SELECT *
    FROM zone_realtime_data
    WHERE (zone_id, timestamp) IN (
      SELECT zone_id, MAX(timestamp)
      FROM zone_realtime_data
      GROUP BY zone_id
    )
  `)
    .then(result => callback(null, result.rows || result))  // sqlite vs pg 호환
    .catch(err => callback(err));
}

function getZoneHistory(zoneId, callback) {
  const query = `
    SELECT
      zone_id,
      strftime('%H', timestamp) AS hour,
      COUNT(*) AS data_points,
      AVG(avg_cycle_time_min) AS avg_cycle_time,
      SUM(ppe_violations) AS total_ppe_violations,
      SUM(hazard_dwell_count) AS total_hazard_dwell
    FROM ZoneRealtimeData
    WHERE zone_id = ?
    GROUP BY hour
    ORDER BY hour
  `;

  db.all(query, [zoneId], (err, rows) => {
    if (err) return callback(err);

    // Unity-friendly JSON 포맷으로 가공
    const formatted = rows.map(row => ({
      hour: row.hour,
      dataPoints: Number(row.data_points),
      avgCycleTime: Number(row.avg_cycle_time.toFixed(2)),
      totalPPEViolations: Number(row.total_ppe_violations),
      totalHazardDwell: Number(row.total_hazard_dwell)
    }));

    callback(null, formatted);
  });
}

module.exports = { getRealtimeZones, getZoneHistory };
