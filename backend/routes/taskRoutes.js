const express = require('express');
const router = express.Router();
const { createTask, getAllTasks, updateTask, deleteTask } = require('../controllers/taskController');

router.post('/', createTask); // POST /tasks
router.get('/', getAllTasks); 
router.put('/:id', updateTask);    // 업무 수정
router.delete('/:id', deleteTask); // 업무 삭제

module.exports = router;
