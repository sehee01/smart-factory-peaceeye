const express = require('express');
const router = express.Router();
const { createTask, getAllTasks, updateTask, deleteTask } = require('../controllers/taskController');

router.post('/tasks', createTask); // POST /tasks
router.get('/tasks', getAllTasks); 
router.put('/tasks/:id', updateTask);    // 업무 수정
router.delete('/tasks/:id', deleteTask); // 업무 삭제

module.exports = router;
