const knex = require('knex');

const db = knex({
  client: 'sqlite3',
  connection: {
    filename: './data.sqlite3', // DB 파일은 backend 폴더 내부에 생성됨
  },
  useNullAsDefault: true,
});

module.exports = db;
