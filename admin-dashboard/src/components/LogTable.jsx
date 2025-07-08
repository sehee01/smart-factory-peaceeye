import React from 'react';
import { DataGrid } from '@mui/x-data-grid';

const cols = [
  { field:'id', headerName:'ID', width:90 },
  { field:'worker', headerName:'Worker', flex:1 },
  { field:'event', headerName:'Event', flex:1 },
  { field:'time', headerName:'Time', flex:1 },
];

export default function LogTable({ rows }) {
  return (
    <div style={{ height: 320, width:'100%' }}>
      <DataGrid rows={rows} columns={cols} pageSize={5} rowsPerPageOptions={[5]} />
    </div>
  );
}

//필요 없는 거