/**
 * LoadingSpinner Component
 * Material-UI based loading animation with progress messages
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  CircularProgress,
  Typography,
  Paper,
  LinearProgress
} from '@mui/material';

export interface LoadingSpinnerProps {
  /** Whether to show the loading spinner */
  loading?: boolean;
}

const loadingMessages = [
  { text: '🔍 Searching for news...', duration: 8000 },
  { text: '📋 Curating sources...', duration: 8000 },
  { text: '✍️  Writing articles...', duration: 8000 },
  { text: '🔎 Critiquing content...', duration: 8000 },
  { text: '🎨 Designing layouts...', duration: 8000 },
  { text: '📰 Compiling newspaper...', duration: 0 }, // Last message stays
];

/**
 * Loading spinner with progress messages
 */
export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  loading = true
}) => {
  const [messageIndex, setMessageIndex] = useState(0);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (!loading) {
      setMessageIndex(0);
      setProgress(0);
      return;
    }

    let timeoutId: NodeJS.Timeout;

    const showNextMessage = () => {
      setMessageIndex((prevIndex) => {
        const nextIndex = prevIndex < loadingMessages.length - 1 ? prevIndex + 1 : prevIndex;
        const duration = loadingMessages[nextIndex].duration;

        if (duration > 0) {
          timeoutId = setTimeout(showNextMessage, duration);
        }

        // Update progress
        setProgress(((nextIndex + 1) / loadingMessages.length) * 100);

        return nextIndex;
      });
    };

    // Start with first message
    const duration = loadingMessages[0].duration;
    if (duration > 0) {
      timeoutId = setTimeout(showNextMessage, duration);
    }

    return () => {
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [loading]);

  if (!loading) return null;

  return (
    <Paper
      elevation={3}
      sx={{
        p: 4,
        mt: 3,
        textAlign: 'center',
        backgroundColor: 'background.paper'
      }}
      role="status"
      aria-live="polite"
      aria-busy="true"
    >
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 3
        }}
      >
        <CircularProgress
          size={60}
          thickness={4}
          sx={{
            color: 'primary.main'
          }}
        />

        <Typography
          variant="h6"
          component="div"
          sx={{
            fontWeight: 500,
            color: 'text.primary',
            minHeight: '2rem'
          }}
        >
          {loadingMessages[messageIndex].text}
        </Typography>

        <Box sx={{ width: '100%', maxWidth: 400 }}>
          <LinearProgress
            variant="determinate"
            value={progress}
            sx={{
              height: 8,
              borderRadius: 4,
              backgroundColor: 'action.hover',
              '& .MuiLinearProgress-bar': {
                borderRadius: 4
              }
            }}
          />
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ mt: 1, display: 'block' }}
          >
            Step {messageIndex + 1} of {loadingMessages.length}
          </Typography>
        </Box>

        <Typography
          variant="body2"
          color="text.secondary"
          sx={{ fontStyle: 'italic' }}
        >
          This may take a few minutes...
        </Typography>
      </Box>
    </Paper>
  );
};
