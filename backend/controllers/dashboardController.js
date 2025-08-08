const db = require("../models/initDB");

const toNumber = (x, d = 0) => (Number.isFinite(+x) ? +x : d);

/** 1) 진행 중 업무 (Unity 최소 DTO)
 *  GET /dashboard/ongoing-tasks
 *  -> { ongoing_tasks: TaskV1[] }
 */
async function getOngoingTasks(req, res) {
  try {
    const { part, delayed, limit, orderBy = "due_date", order = "asc" } = req.query;
    const q = db("ongoing_tasks").select(
      "task_name", "part", "due_date",
      db.raw("CAST(COALESCE(progress, 0) AS INTEGER) AS progress"),
      db.raw("CAST(COALESCE(is_delayed, 0) AS INTEGER) AS is_delayed")
    );

    if (part) q.where("part", part);
    if (delayed === "1" || delayed === "0") q.where("is_delayed", delayed === "1" ? 1 : 0);

    const ALLOWED_ORDER_BY = new Set(["due_date","progress","task_name","part","is_delayed"]);
    q.orderBy(ALLOWED_ORDER_BY.has(orderBy) ? orderBy : "due_date", (order === "desc" ? "desc" : "asc"));

    if (limit) q.limit(Math.max(1, Math.min(100, parseInt(limit))));

    const rows = await q;
    const ongoing_tasks = rows.map(r => ({
      task_name: r.task_name ?? "",
      part: r.part ?? "",
      due_date: r.due_date ?? "",
      progress: Number(r.progress) || 0,
      is_delayed: !!Number(r.is_delayed),
    }));

    res.json({ ongoing_tasks });
  } catch (err) {
    console.error("[ongoing-tasks ERROR]", err);
    res.status(500).json({ error: "ONGOING_TASKS_QUERY_FAILED" });
  }
}

/** 2) 파트별 요약 (+ 팀 오버뷰 모드)
 *  GET /dashboard/part-summary
 *    - 기본: 기존 파트 요약 (ongoing_tasks 기반)
 *    - 팀 모드: /dashboard/part-summary?by=team[&rateBase=members|active][&alertHours=24]
 *        응답: { part_summary: [{ part: team_id, task_count: members, delay_rate, active_today }] }
 */
async function getPartSummary(req, res) {
  try {
    const by = (req.query.by || "").toLowerCase();

    if (by === "team") {
      // ---- 팀 오버뷰 모드 ----
      const rateBase = (req.query.rateBase || "members").toLowerCase(); // 'members' | 'active'
      const alertHours = toNumber(req.query.alertHours, 24);

      // 오늘 00:00 UTC
      const now = new Date();
      const startOfUTCDay = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate())).toISOString();
      // 알림 기준 시각
      const sinceAlertsISO = new Date(Date.now() - alertHours * 3600 * 1000).toISOString();

      // 1) 팀 인원수 (workers)
      const membersRows = await db("workers")
        .select("team_id")
        .count({ members: "worker_id" })
        .groupBy("team_id");

      const membersMap = new Map();
      for (const r of membersRows) membersMap.set(r.team_id ?? "UNKNOWN", toNumber(r.members));

      // 2) 오늘 활동한 작업자 (worker_details) → distinct worker_id를 team으로 매핑
      const activeWorkerIds = db("worker_details")
        .distinct("worker_id")
        .where("timestamp", ">", startOfUTCDay);

      const activeRows = await db("workers as w")
        .join(activeWorkerIds.as("a"), "w.worker_id", "a.worker_id")
        .select("w.team_id")
        .count({ active_today: "w.worker_id" })
        .groupBy("w.team_id");

      const activeMap = new Map();
      for (const r of activeRows) activeMap.set(r.team_id ?? "UNKNOWN", toNumber(r.active_today));

      // 3) 최근 알림 보유 작업자 수(=지연율의 분자 대용). 스키마: worker_alerts(worker_id, zone_id, timestamp, violations)
      //    "알림이 있는 작업자"를 distinct로 집계
      const alertedWorkerIds = db("worker_alerts")
        .select("worker_id")
        .where("timestamp", ">", sinceAlertsISO)
        .groupBy("worker_id");

      const alertRows = await db("workers as w")
        .join(alertedWorkerIds.as("aw"), "w.worker_id", "aw.worker_id")
        .select("w.team_id")
        .countDistinct({ alerted: "w.worker_id" })
        .groupBy("w.team_id");

      const alertedMap = new Map();
      for (const r of alertRows) alertedMap.set(r.team_id ?? "UNKNOWN", toNumber(r.alerted));

      // 4) 합치기
      const teamIds = new Set([...membersMap.keys(), ...activeMap.keys(), ...alertedMap.keys()]);
      const part_summary = [];

      for (const team_id of teamIds) {
        const members = membersMap.get(team_id) || 0;
        const active_today = activeMap.get(team_id) || 0;
        const alerted = alertedMap.get(team_id) || 0;

        const base = rateBase === "active" ? active_today : members;
        const delay_rate = base > 0 ? alerted / base : 0; // 0~1

        part_summary.push({
          part: team_id,           // ← 기존 DTO의 part 필드를 team_id로 사용
          task_count: members,     // ← members
          delay_rate,              // ← 0~1
          active_today,            // ← 추가 필드 (클라 DTO에 하나 추가해두면 됨)
        });
      }

      // 정렬: active_today desc → members desc
      part_summary.sort((a, b) => (b.active_today - a.active_today) || (b.task_count - a.task_count));

      return res.json({ part_summary });
    }

    // ---- 기존 파트 요약(그대로) ----
    const rows = await db("ongoing_tasks")
      .select("part")
      .count({ task_count: "id" })
      .avg({ delay_rate: "is_delayed" })
      .groupBy("part");

    const part_summary = rows.map(r => ({
      part: r.part,
      task_count: toNumber(r.task_count),
      delay_rate: toNumber(r.delay_rate), // 0~1
    }));

    res.json({ part_summary });
  } catch (err) {
    console.error("[part-summary ERROR]", err);
    res.status(500).json({ error: "PART_SUMMARY_QUERY_FAILED" });
  }
}


/** 3) 작업자 상태
 *  GET /dashboard/worker-status
 *  기본: [{ worker_id, total_tasks }]
 *  상세: /dashboard/worker-status?detail=1[&hours=24]
 *        -> [{ worker_id, name, team_id, total_tasks, active_today, zone_id, product_count, alerts }]
 */
async function getWorkerStatus(req, res) {
  try {
    const detail = req.query.detail === "1";
    if (!detail) {
      // 기존 동작 (변경 없음)
      const rows = await db("worker_details")
        .select("worker_id")
        .count({ total_tasks: "id" })
        .groupBy("worker_id");

      const worker_status = rows.map(r => ({
        worker_id: r.worker_id,
        total_tasks: Number(r.total_tasks) || 0,
      }));
      return res.json({ worker_status });
    }

    // ── detail=1일 때 확장 응답 ──
    const hours = Number.isFinite(+req.query.hours) ? +req.query.hours : 24;
    const dayStartISO = new Date(new Date().toISOString().slice(0,10) + "T00:00:00.000Z").toISOString();
    const sinceISO = new Date(Date.now() - hours * 3600 * 1000).toISOString();

    // 전체 태스크 수(기존)
    const totals = await db("worker_details")
      .select("worker_id")
      .count({ total_tasks: "id" })
      .groupBy("worker_id");

    const totalMap = new Map(totals.map(r => [r.worker_id, Number(r.total_tasks) || 0]));

    // 오늘 활성(스냅샷 있는 작업자)
    const activeRows = await db("worker_details")
      .distinct("worker_id")
      .where("timestamp", ">", dayStartISO);
    const activeSet = new Set(activeRows.map(r => r.worker_id));

    // 각 작업자의 최신 스냅샷 (zone_id, product_count)
    const latestTs = await db("worker_details as wd")
      .select("worker_id")
      .max({ ts: "timestamp" })
      .groupBy("worker_id");
    let latestRows = [];
    if (latestTs.length) {
      latestRows = await db("worker_details as wd")
        .whereIn(db.raw("(wd.worker_id, wd.timestamp)"),
          latestTs.map(r => [r.worker_id, r.ts]))
        .select("wd.worker_id", "wd.zone_id", "wd.product_count");
    }
    const latestMap = new Map(latestRows.map(r => [r.worker_id, r]));

    // 알림 수 (최근 hours)
    const alerts = await db("worker_alerts")
      .select("worker_id")
      .count({ alerts: "*" })
      .where("timestamp", ">", sinceISO)
      .groupBy("worker_id");
    const alertMap = new Map(alerts.map(r => [r.worker_id, Number(r.alerts) || 0]));

    // 기본 인적 정보 (이름/팀)
    const people = await db("workers").select("worker_id", "name", "team_id");
    const peopleMap = new Map(people.map(r => [r.worker_id, r]));

    // 머지
    const allWorkerIds = new Set([
      ...peopleMap.keys(),
      ...totalMap.keys(),
      ...latestMap.keys(),
      ...alertMap.keys()
    ]);

    const worker_status = [...allWorkerIds].map(id => {
      const p = peopleMap.get(id) || {};
      const last = latestMap.get(id) || {};
      return {
        worker_id: id,
        name: p.name || "",
        team_id: p.team_id || "",
        total_tasks: totalMap.get(id) || 0,
        active_today: activeSet.has(id),
        zone_id: last.zone_id || "",
        product_count: Number(last.product_count) || 0,
        alerts: alertMap.get(id) || 0,
      };
    })
    // 기본 정렬: active desc → alerts desc → name asc
    .sort((a,b) => (b.active_today - a.active_today) || (b.alerts - a.alerts) || String(a.name).localeCompare(b.name));

    res.json({ worker_status, hours });
  } catch (err) {
    console.error("[worker-status ERROR]", err);
    res.status(500).json({ error: "WORKER_STATUS_QUERY_FAILED" });
  }
}

/** 4) 위험 요약
 *  GET /dashboard/risk-summary
 *  스키마:
 *    workers(worker_id, name, team_id, position)
 *    zone_realtime_data(zone_name, zone_type, timestamp, active_workers, active_tasks, avg_cycle_time_min, ppe_violations, hazard_dwell_count)
 *  정의(가정):
 *    - total_workers: workers의 고유 worker_id 수
 *    - ppe_violation_rate(%): 최근 24h의 모든 zone ppe_violations 합 / total_workers * 100
 *    - roi_violation_count: 최근 24h의 모든 zone hazard_dwell_count 합
 */
async function getRiskSummary(req, res) {
  try {
    const dayAgoISO = new Date(Date.now() - 24 * 3600 * 1000).toISOString();

    // team_id 별 인원수
    const workersByTeam = await db("workers")
      .select("team_id")
      .count({ total_workers: "worker_id" })
      .groupBy("team_id");

    const teamCountMap = new Map(
      workersByTeam.map(r => [r.team_id ?? "UNKNOWN", Number(r.total_workers) || 0])
    );

    // 최근 24h 알림: 팀별 집계
    const alertsByTeam = await db("worker_alerts as wa")
      .join("workers as w", "w.worker_id", "wa.worker_id")
      .where("wa.timestamp", ">", dayAgoISO)
      .select(
        "w.team_id",
        db.raw("COUNT(*) as total_alerts"),
        db.raw("COUNT(DISTINCT wa.worker_id) as workers_with_alerts")
      )
      .groupBy("w.team_id");

    const alertAggMap = new Map(
      alertsByTeam.map(r => [
        r.team_id ?? "UNKNOWN",
        {
          total_alerts: Number(r.total_alerts) || 0,
          workers_with_alerts: Number(r.workers_with_alerts) || 0,
        },
      ])
    );

    // 결과 구성
    const teamIds = new Set([...teamCountMap.keys(), ...alertAggMap.keys()]);
    const risk_summary_by_team = [];

    for (const team_id of teamIds) {
      const total_workers = teamCountMap.get(team_id) || 0;
      const agg = alertAggMap.get(team_id) || { total_alerts: 0, workers_with_alerts: 0 };

      const ppe_violation_rate =
        total_workers > 0 ? Number(((agg.workers_with_alerts / total_workers) * 100).toFixed(1)) : 0;

      risk_summary_by_team.push({
        team_id,
        total_workers,
        ppe_violation_rate,
        roi_violation_count: agg.total_alerts,
      });
    }

    // 정렬
    risk_summary_by_team.sort(
      (a, b) => b.ppe_violation_rate - a.ppe_violation_rate || b.total_workers - a.total_workers
    );

    return res.json({ risk_summary_by_team });

  } catch (err) {
    console.error("[risk-summary-teams ERROR]", err);
    res.status(500).json({ error: "RISK_SUMMARY_TEAMS_QUERY_FAILED" });
  }
}

// 5) CSV 유틸: 값 이스케이프
function csvEscape(v) {
  if (v == null) return "";
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}
function toCSV(headers, rows) {
  const head = headers.map(csvEscape).join(",") + "\n";
  const body = rows.map(r => headers.map(h => csvEscape(r[h])).join(",")).join("\n");
  return head + body + "\n";
}

/**
 * GET /dashboard/export/csv?kind=... + 추가 파라미터
 * kind:
 *   - ongoing            -> ongoing_tasks
 *   - part-team          -> /dashboard/part-summary?by=team 형태 데이터
 *   - worker-status      -> /dashboard/worker-status?detail=1
 *   - risk-teams         -> /dashboard/risk-summary?by=teams [&hours=24]
 */
function csvEscape(v) {
  if (v == null) return "";
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}
function toCSV(headers, rows) {
  const head = headers.join(",") + "\n";
  const body = rows.map(r => headers.map(h => csvEscape(r[h])).join(",")).join("\n");
  return head + body + "\n";
}

/**
 * GET /dashboard/export/combined?hours=24
 * 한 CSV 파일에 4가지 섹션을 합쳐서 반환
 * - dataset: ongoing | team | worker | risk
 * - 컬럼은 접두사로 구분(ong_, team_, wrk_, risk_)
 */
async function exportCsv(req, res) {
  try {
    const hours = Number.isFinite(+req.query.hours) ? +req.query.hours : 24;
    const sinceISO = new Date(Date.now() - hours * 3600 * 1000).toISOString();
    const dayStartISO = new Date(new Date().toISOString().slice(0,10) + "T00:00:00.000Z").toISOString();

    // 공통 준비
    // workers(team_id 집계)
    const workersByTeam = await db("workers")
      .select("team_id")
      .count({ total_workers: "worker_id" })
      .groupBy("team_id");
    const teamCountMap = new Map(workersByTeam.map(r => [r.team_id ?? "UNKNOWN", Number(r.total_workers) || 0]));

    // =========================
    // 1) ongoing_tasks → dataset=ongoing
    // =========================
    const ongoing = await db("ongoing_tasks")
      .select("task_name","part","due_date","progress","is_delayed")
      .orderBy("due_date","asc");

    // =========================
    // 2) team overview → dataset=team
    // (part=team_id, task_count=members, active_today, delay_rate[0~1])
    // =========================
    // active_today
    const activeWorkerIds = db("worker_details").distinct("worker_id").where("timestamp", ">", dayStartISO);
    const activeRows = await db("workers as w")
      .join(activeWorkerIds.as("a"), "w.worker_id", "a.worker_id")
      .select("w.team_id")
      .count({ active_today: "w.worker_id" })
      .groupBy("w.team_id");
    const activeMap = new Map(activeRows.map(r => [r.team_id ?? "UNKNOWN", Number(r.active_today) || 0]));

    // alerted workers (최근 hours)
    const alertedWorkerIds = db("worker_alerts")
      .select("worker_id")
      .where("timestamp", ">", sinceISO)
      .groupBy("worker_id");
    const alertRowsTeam = await db("workers as w")
      .join(alertedWorkerIds.as("aw"), "w.worker_id", "aw.worker_id")
      .select("w.team_id")
      .countDistinct({ alerted: "w.worker_id" })
      .groupBy("w.team_id");
    const alertedMap = new Map(alertRowsTeam.map(r => [r.team_id ?? "UNKNOWN", Number(r.alerted) || 0]));

    const teamIds = new Set([...teamCountMap.keys(), ...activeMap.keys(), ...alertedMap.keys()]);
    const teamOverview = [...teamIds].map(team_id => {
      const members = teamCountMap.get(team_id) || 0;
      const active_today = activeMap.get(team_id) || 0;
      const alerted = alertedMap.get(team_id) || 0;
      const delay_rate = members > 0 ? alerted / members : 0; // 0~1
      return { part: team_id, task_count: members, active_today, delay_rate };
    });

    // =========================
    // 3) worker status(detail) → dataset=worker
    // =========================
    const totals = await db("worker_details")
      .select("worker_id")
      .count({ total_tasks: "id" })
      .groupBy("worker_id");
    const totalMap = new Map(totals.map(r => [r.worker_id, Number(r.total_tasks) || 0]));

    const activeSet = new Set((await db("worker_details").distinct("worker_id").where("timestamp",">",dayStartISO))
      .map(r => r.worker_id));

    const latestTs = await db("worker_details as wd")
      .select("worker_id")
      .max({ ts: "timestamp" })
      .groupBy("worker_id");
    let latestRows = [];
    if (latestTs.length) {
      latestRows = await db("worker_details as wd")
        .whereIn(db.raw("(wd.worker_id, wd.timestamp)"), latestTs.map(r => [r.worker_id, r.ts]))
        .select("wd.worker_id","wd.zone_id","wd.product_count");
    }
    const latestMap = new Map(latestRows.map(r => [r.worker_id, r]));
    const alertsByWorker = await db("worker_alerts")
      .select("worker_id")
      .count({ alerts: "*" })
      .where("timestamp", ">", sinceISO)
      .groupBy("worker_id");
    const alertMap = new Map(alertsByWorker.map(r => [r.worker_id, Number(r.alerts) || 0]));
    const people = await db("workers").select("worker_id","name","team_id");

    const workers = people.map(p => {
      const last = latestMap.get(p.worker_id) || {};
      return {
        worker_id: p.worker_id,
        name: p.name || "",
        team_id: p.team_id || "",
        total_tasks: totalMap.get(p.worker_id) || 0,
        active_today: activeSet.has(p.worker_id) ? 1 : 0,
        zone_id: last.zone_id || "",
        product_count: Number(last.product_count) || 0,
        alerts: alertMap.get(p.worker_id) || 0,
      };
    });

    // =========================
    // 4) risk summary by team → dataset=risk
    // =========================
    const alertsByTeam = await db("worker_alerts as wa")
      .join("workers as w", "w.worker_id", "wa.worker_id")
      .where("wa.timestamp", ">", sinceISO)
      .select(
        "w.team_id",
        db.raw("COUNT(*) as total_alerts"),
        db.raw("COUNT(DISTINCT wa.worker_id) as workers_with_alerts")
      )
      .groupBy("w.team_id");

    const risks = alertsByTeam.map(r => {
      const team_id = r.team_id ?? "UNKNOWN";
      const total_workers = teamCountMap.get(team_id) || 0;
      const workers_with_alerts = Number(r.workers_with_alerts) || 0;
      const ppe_violation_rate = total_workers > 0
        ? Number(((workers_with_alerts / total_workers) * 100).toFixed(1))
        : 0;
      return {
        team_id,
        total_workers,
        ppe_violation_rate,
        roi_violation_count: Number(r.total_alerts) || 0,
      };
    });

    // =========================
    // 하나의 CSV로 합치기
    // =========================
    const headers = [
      "dataset",
      // ongoing (ong_)
      "ong_task_name","ong_part","ong_due_date","ong_progress","ong_is_delayed",
      // team overview (team_)
      "team_part","team_task_count","team_active_today","team_delay_rate",
      // worker (wrk_)
      "wrk_worker_id","wrk_name","wrk_team_id","wrk_total_tasks","wrk_active_today","wrk_zone_id","wrk_product_count","wrk_alerts",
      // risk (risk_)
      "risk_team_id","risk_total_workers","risk_ppe_violation_rate","risk_roi_violation_count"
    ];

    const rows = [];

    // ongoing
    for (const r of ongoing) {
      rows.push({
        dataset: "ongoing",
        ong_task_name: r.task_name,
        ong_part: r.part,
        ong_due_date: r.due_date,
        ong_progress: r.progress,
        ong_is_delayed: r.is_delayed,
      });
    }

    // team overview
    for (const r of teamOverview) {
      rows.push({
        dataset: "team",
        team_part: r.part,
        team_task_count: r.task_count,
        team_active_today: r.active_today,
        team_delay_rate: r.delay_rate, // 0~1
      });
    }

    // worker
    for (const w of workers) {
      rows.push({
        dataset: "worker",
        wrk_worker_id: w.worker_id,
        wrk_name: w.name,
        wrk_team_id: w.team_id,
        wrk_total_tasks: w.total_tasks,
        wrk_active_today: w.active_today,
        wrk_zone_id: w.zone_id,
        wrk_product_count: w.product_count,
        wrk_alerts: w.alerts,
      });
    }

    // risk
    for (const r of risks) {
      rows.push({
        dataset: "risk",
        risk_team_id: r.team_id,
        risk_total_workers: r.total_workers,
        risk_ppe_violation_rate: r.ppe_violation_rate, // %
        risk_roi_violation_count: r.roi_violation_count,
      });
    }

    const csv = toCSV(headers, rows);
    res.setHeader("Content-Type", "text/csv; charset=utf-8");
    res.setHeader("Content-Disposition", `attachment; filename=dashboard_all_${hours}h.csv`);
    return res.send(csv);
  } catch (err) {
    console.error("[export-combined ERROR]", err);
    res.status(500).json({ error: "CSV_COMBINED_EXPORT_FAILED" });
  }
}

module.exports = {
  getOngoingTasks,
  getPartSummary,
  getWorkerStatus,
  getRiskSummary,
  exportCsv
};