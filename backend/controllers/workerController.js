const { getWorkerHistoryFromDB } = require("../services/workerQueryService");

async function getWorkerHistory(req, res) {
  const workerId = req.params.worker_id;

  try {
    const result = await getWorkerHistoryFromDB(workerId);
    res.json(result);
  } catch (err) {
    console.error("[WorkerHistory ERROR]", err.message);
    res.status(500).json({ error: "DB 조회 실패" });
  }
}

module.exports = { getWorkerHistory };
