const db = require("../models/initDB");

async function getDashboardSummary(req, res) {
  try {
    // [1] 진행 중인 업무
    const ongoing_tasks = await db("ongoing_tasks").select("*");

    // [2] 파트별 요약
    const part_summary = await db("ongoing_tasks")
      .select("part")
      .count("id as task_count")
      .avg("is_delayed as delay_rate")
      .groupBy("part");

    // [3] 작업자별 상태
    const worker_status = await db("worker_details")
      .select("worker_id")
      .count("id as total_tasks")
      .groupBy("worker_id");

    // 최근 3일 기준 경고 내역 (예시: 최근 72시간)
    const now = new Date();
    const threeDaysAgo = new Date(now.getTime() - 3 * 24 * 60 * 60 * 1000).toISOString();
    const recent_alerts = await db("worker_alerts")
      .select("worker_id", "type", db.raw("COUNT(*) as count"))
      .where("timestamp", ">", threeDaysAgo)
      .groupBy("worker_id", "type");

    // [4] 누적 위험 요약
    const totalWorkers = await db("worker_details").countDistinct("worker_id as count");
    const totalPPEViolations = await db("worker_alerts").where("type", "ppe_violation").count("* as count");
    const totalROIViolations = await db("worker_alerts").where("type", "roi_violation").count("* as count");

    const risk_summary = {
      total_workers: totalWorkers[0].count,
      ppe_violation_rate: `${((totalPPEViolations[0].count / totalWorkers[0].count) * 100).toFixed(1)}%`,
      roi_violation_count: totalROIViolations[0].count
    };

    res.json({
      ongoing_tasks,
      part_summary,
      worker_status,
      recent_alerts,
      risk_summary
    });
  } catch (err) {
    console.error("[Dashboard Summary ERROR]", err.message);
    res.status(500).json({ error: "Dashboard 요약 로딩 실패" });
  }
}

module.exports = { getDashboardSummary };
