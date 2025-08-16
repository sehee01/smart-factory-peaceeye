const db = require("../models/initDB");

async function saveToSQLite(data) {
  const trx = await db.transaction();

  try {
    const now = new Date().toISOString();

    // [1] workers 저장
    if (Array.isArray(data.workers)) {
      for (const w of data.workers) {
        // 필수 필드 검증
        if (!w.worker_id || w.x === undefined || w.y === undefined) {
          console.warn("[DB WARNING] Skipping worker with missing data:", w);
          continue;
        }
        
        await trx("worker_details").insert({
          worker_id: w.worker_id,
          zone_id: w.zone_id || "Z00",
          x: w.x,
          y: w.y,
          product_count: w.product_count || 0,
          timestamp: w.timestamp || now,
        });
      }
    }

    // [2] zones 저장
    if (Array.isArray(data.zones)) {
      for (const z of data.zones) {
        await trx("zone_realtime_data").insert({
          zone_id: z.zone_id,
          zone_name: z.zone_name,
          zone_type: z.zone_type,
          timestamp: data.timestamp || now,
          active_workers: z.active_workers,
          active_tasks: z.active_tasks,
          avg_cycle_time_min: z.avg_cycle_time_min,
          ppe_violations: z.ppe_violations,
          hazard_dwell_count: z.hazard_dwell_count,
        });
      }
    }

    // [3] violations 저장 (전체 JSON 통째로 string 저장)
    if (Array.isArray(data.violations)) {
      for (const v of data.violations) {
        await trx("worker_alerts").insert({
          worker_id: v.worker_id,
          zone_id: v.zone_id,
          timestamp: data.timestamp || now,
          violations: JSON.stringify(v.violations),  // ✅ 전체 violations JSON을 문자열로 저장
        });
      }
    }

    await trx.commit();
    console.log("[DB] 데이터 저장 완료");
  } catch (err) {
    await trx.rollback();
    console.error("[DB ERROR]", err);
  }
}

module.exports = saveToSQLite;
