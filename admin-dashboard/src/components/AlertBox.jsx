import React from 'react';
import { Snackbar, Alert } from '@mui/material';

export default function AlertBox({ open, onClose, message, severity='warning' }) {
  return (
    <Snackbar open={open} autoHideDuration={4000} onClose={onClose} anchorOrigin={{ vertical:'top', horizontal:'right' }}>
      <Alert onClose={onClose} severity={severity} variant="filled">{message}</Alert>
    </Snackbar>
  );
}

//필요 없는 거