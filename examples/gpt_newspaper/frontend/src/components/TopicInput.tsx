/**
 * TopicInput Component
 * Material-UI based topic input field with add/remove functionality
 */

import React from 'react';
import {
  Box,
  TextField,
  IconButton,
  Tooltip
} from '@mui/material';
import AddCircleIcon from '@mui/icons-material/AddCircle';
import RemoveCircleIcon from '@mui/icons-material/RemoveCircle';

export interface TopicInputProps {
  /** Unique identifier for the topic */
  id: number;
  /** Current value of the input */
  value: string;
  /** Callback when value changes */
  onChange: (id: number, value: string) => void;
  /** Callback to add a new topic */
  onAdd: () => void;
  /** Callback to remove this topic */
  onRemove: (id: number) => void;
  /** Whether this is the first topic (cannot be removed) */
  isFirst: boolean;
  /** Whether this is the last topic (shows add/remove buttons) */
  isLast: boolean;
  /** Whether more topics can be added */
  canAddMore: boolean;
  /** Whether the input is disabled */
  disabled?: boolean;
}

/**
 * Topic input component with add/remove controls
 */
export const TopicInput: React.FC<TopicInputProps> = ({
  id,
  value,
  onChange,
  onAdd,
  onRemove,
  isFirst,
  isLast,
  canAddMore,
  disabled = false,
}) => {
  const placeholders = [
    'e.g., AI developments',
    'e.g., Climate updates',
    'e.g., Technology trends',
    'e.g., Business news',
    'e.g., Science discoveries'
  ];

  const placeholder = placeholders[(id - 1) % placeholders.length];

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 1,
        mb: 2
      }}
    >
      <TextField
        fullWidth
        variant="outlined"
        value={value}
        onChange={(e) => onChange(id, e.target.value)}
        placeholder={placeholder}
        disabled={disabled}
        required
        inputProps={{
          'aria-label': `Topic ${id}`,
          'aria-required': 'true'
        }}
        sx={{
          '& .MuiOutlinedInput-root': {
            '&:hover fieldset': {
              borderColor: 'primary.main',
            },
          },
        }}
      />
      {isLast && (
        <Box sx={{ display: 'flex', gap: 0.5 }}>
          {canAddMore && (
            <Tooltip title="Add topic" arrow>
              <IconButton
                onClick={onAdd}
                color="success"
                disabled={disabled}
                aria-label="Add topic"
                size="large"
              >
                <AddCircleIcon fontSize="large" />
              </IconButton>
            </Tooltip>
          )}
          {!isFirst && (
            <Tooltip title="Remove topic" arrow>
              <IconButton
                onClick={() => onRemove(id)}
                color="error"
                disabled={disabled}
                aria-label="Remove topic"
                size="large"
              >
                <RemoveCircleIcon fontSize="large" />
              </IconButton>
            </Tooltip>
          )}
        </Box>
      )}
    </Box>
  );
};
