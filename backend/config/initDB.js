const db = require('./dbConnect');

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

// detection_results 테이블 생성
db.schema.hasTable('detection_results').then((exists) => {
  if (!exists) {
    return db.schema.createTable('detection_results', (table) => {
      table.increments('id').primary();
      table.string('timestamp').notNullable();
    }).then(() => {
      console.log("[detection_results] 테이블 생성 완료");
    });
  }
});

// workers 테이블 생성 (1:N 관계로 연결)
db.schema.hasTable('workers').then((exists) => {
  if (!exists) {
    return db.schema.createTable('workers', (table) => {
      table.increments('id').primary();
      table.integer('detection_result_id').unsigned().references('id').inTable('detection_results').onDelete('CASCADE');
      table.string('worker_id').notNullable();
      table.float('x').notNullable();
      table.float('y').notNullable();
      table.enu('status', ['normal', 'warning', 'danger']).defaultTo('normal');
    }).then(() => {
      console.log("[workers] 테이블 생성 완료");
    });
  }
});

