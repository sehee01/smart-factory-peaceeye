const db = require("../config/dbConnect");

async function getWorkerHistoryFromDB(workerId) {
  return await db("worker_details")
    .where("worker_id", workerId)
    .orderBy("timestamp", "desc");
}

module.exports = { getWorkerHistoryFromDB };
