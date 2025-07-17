// models/DetectionResult.js
const mongoose = require('mongoose');

const detectionSchema = new mongoose.Schema({
  timestamp: {
    type: String,
    required: true,
  },
  workers: [
    {
      worker_id: {
        type: String,
        required: true,
      },
      x: {
        type: Number,
        required: true,
      },
      y: {
        type: Number,
        required: true,
      },
      status: {
        type: String,
        enum: ['normal', 'warning', 'danger'],
        default: 'normal',
      }
    }
  ]
});

module.exports = mongoose.model('DetectionResult', detectionSchema);
