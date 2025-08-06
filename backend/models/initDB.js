const db = require('../config/dbConnect');

// Admin 테이블 생성
db.schema.hasTable('admins').then((exists) => {
  if (!exists) {
    return db.schema.createTable('admins', (table) => {
      table.increments('id').primary();
      table.string('username').unique().notNullable();
      table.string('password').notNullable();
      table.timestamps(true, true); // created_at, updated_at 자동
    }).then(() => {
      console.log("[admins] 테이블 생성 완료");
    });
  }
});

// [zone_realtime_data] 테이블 생성
db.schema.hasTable('zone_realtime_data').then((exists) => {
  if (!exists) {
    return db.schema.createTable('zone_realtime_data', (table) => {
      table.increments('id').primary();
      table.string('zone_id').notNullable();
      table.string('zone_name');
      table.string('zone_type');
      table.string('timestamp').notNullable();
      table.integer('active_workers');
      table.string('active_tasks');
      table.float('avg_cycle_time_min');
      table.integer('ppe_violations');
      table.integer('hazard_dwell_count');
    }).then(() => {
      console.log("[zone_realtime_data] 테이블 생성 완료");
    });
  }
});

// [worker_details] 테이블 생성
db.schema.hasTable('worker_details').then((exists) => {
  if (!exists) {
    return db.schema.createTable('worker_details', (table) => {
      table.increments('id').primary();
      table.string('worker_id').notNullable();
      table.string('zone_id');
      table.float('x');
      table.float('y');
      table.integer('product_count');
      table.string('timestamp').notNullable();
    }).then(() => console.log("[worker_details] 테이블 생성 완료"));
  }
});

// [worker_alerts] 테이블 생성 (violations 전체 JSON 저장용)
db.schema.hasTable('worker_alerts').then((exists) => {
  if (!exists) {
    return db.schema.createTable('worker_alerts', (table) => {
      table.increments('id').primary();
      table.string('worker_id').notNullable();
      table.string('zone_id').notNullable();
      table.string('timestamp').notNullable();
      table.text('violations').notNullable(); // ← JSON string 전체 저장
    }).then(() => {
      console.log("[worker_alerts] 테이블 생성 완료 (with JSON violations)");
    });
  }
});

module.exports = db;