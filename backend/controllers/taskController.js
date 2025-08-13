const db = require('../models/initDB'); // initDB에서 knex 인스턴스를 export 했다고 가정

async function createTask(req, res) {
  const { task_name, part, due_date, details } = req.body;

  if (!task_name || !part || !due_date || !details) {
    return res.status(400).json({ error: "모든 필드를 입력해주세요." });
  }

  try {
    const [insertedId] = await db('ongoing_tasks').insert({
      task_name,
      part,
      due_date,
      details,
      progress: 0,
      is_delayed: false,
    });

    res.status(201).json({ message: "업무가 추가되었습니다.", task_id: insertedId });
  } catch (err) {
    console.error("[createTask ERROR]", err.message);
    res.status(500).json({ error: "DB 오류로 인해 저장 실패" });
  }
}

async function getAllTasks(req, res) {
  try {
    const tasks = await db('ongoing_tasks').select('*');
    res.status(200).json(tasks);
  } catch (err) {
    console.error("[getAllTasks ERROR]", err.message);
    res.status(500).json({ error: "DB에서 업무를 불러오는 데 실패했습니다." });
  }
}

// 업무 수정
async function updateTask(req, res) {
  const { id } = req.params;
  const { task_name, part, due_date, details, progress, is_delayed } = req.body;

  try {
    await db('ongoing_tasks').where({ id }).update({
      task_name,
      part,
      due_date,
      details,
      progress,
      is_delayed,
    });
    res.status(200).json({ message: "업무가 수정되었습니다." });
  } catch (err) {
    console.error("[updateTask ERROR]", err.message);
    res.status(500).json({ error: "업무 수정 실패" });
  }
}

// 업무 삭제
async function deleteTask(req, res) {
  const { id } = req.params;

  try {
    await db('ongoing_tasks').where({ id }).del();
    res.status(200).json({ message: "업무가 삭제되었습니다." });
  } catch (err) {
    console.error("[deleteTask ERROR]", err.message);
    res.status(500).json({ error: "업무 삭제 실패" });
  }
}


module.exports = {
  createTask,
  getAllTasks,
  updateTask,
  deleteTask
};
