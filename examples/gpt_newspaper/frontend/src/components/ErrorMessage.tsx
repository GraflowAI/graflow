/**
 * ErrorMessage Component
 * Material-UI based error alert with dismiss functionality
 */

import React from 'react';
import {
  Alert,
  AlertTitle,
  IconButton,
  Collapse
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

export interface ErrorMessageProps {
  /** Error message to display */
  message: string;
  /** Optional error title */
  title?: string;
  /** Callback when dismiss button is clicked */
  onDismiss?: () => void;
  /** Severity level */
  severity?: 'error' | 'warning' | 'info';
  /** Whether to show the error */
  open?: boolean;
}

/**
 * Error message alert component
 */
export const ErrorMessage: React.FC<ErrorMessageProps> = ({
  message,
  title,
  onDismiss,
  severity = 'error',
  open = true
}) => {
  return (
    <Collapse in={open}>
      <Alert
        severity={severity}
        sx={{
          mt: 2,
          mb: 2
        }}
        action={
          onDismiss && (
            <IconButton
              aria-label="close"
              color="inherit"
              size="small"
              onClick={onDismiss}
            >
              <CloseIcon fontSize="inherit" />
            </IconButton>
          )
        }
        role="alert"
        aria-live="assertive"
      >
        {title && <AlertTitle>{title}</AlertTitle>}
        {message}
      </Alert>
    </Collapse>
  );
};
