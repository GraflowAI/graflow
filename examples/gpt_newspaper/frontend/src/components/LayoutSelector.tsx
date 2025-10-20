/**
 * LayoutSelector Component
 * Material-UI based newspaper layout selector
 */

import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Radio,
  RadioGroup,
  FormControlLabel,
  FormControl,
  FormLabel
} from '@mui/material';
import ViewColumnIcon from '@mui/icons-material/ViewColumn';
import ViewQuiltIcon from '@mui/icons-material/ViewQuilt';
import ViewCompactIcon from '@mui/icons-material/ViewCompact';
import type { LayoutOption } from '../types';

export interface LayoutSelectorProps {
  /** Currently selected layout */
  selected: LayoutOption;
  /** Callback when layout changes */
  onChange: (layout: LayoutOption) => void;
  /** Whether the selector is disabled */
  disabled?: boolean;
}

interface LayoutConfig {
  id: LayoutOption;
  name: string;
  description: string;
  icon: React.ReactNode;
}

const layouts: LayoutConfig[] = [
  {
    id: 'layout_1.html',
    name: 'Classic',
    description: 'Traditional newspaper layout',
    icon: <ViewColumnIcon fontSize="large" />
  },
  {
    id: 'layout_2.html',
    name: 'Modern',
    description: 'Contemporary grid design',
    icon: <ViewQuiltIcon fontSize="large" />
  },
  {
    id: 'layout_3.html',
    name: 'Compact',
    description: 'Space-efficient layout',
    icon: <ViewCompactIcon fontSize="large" />
  },
];

/**
 * Layout selector component
 */
export const LayoutSelector: React.FC<LayoutSelectorProps> = ({
  selected,
  onChange,
  disabled = false,
}) => {
  return (
    <FormControl component="fieldset" fullWidth disabled={disabled}>
      <FormLabel
        component="legend"
        sx={{
          fontSize: '1.1rem',
          fontWeight: 600,
          color: 'text.primary',
          mb: 2
        }}
      >
        Select Newspaper Layout
      </FormLabel>
      <RadioGroup
        aria-label="newspaper layout"
        value={selected}
        onChange={(e) => onChange(e.target.value as LayoutOption)}
        sx={{
          display: 'flex',
          flexDirection: 'row',
          gap: 2,
          flexWrap: 'wrap'
        }}
      >
        {layouts.map((layout) => (
          <Card
            key={layout.id}
            sx={{
              flex: '1 1 200px',
              minWidth: 200,
              maxWidth: 250,
              cursor: disabled ? 'not-allowed' : 'pointer',
              border: 2,
              borderColor: selected === layout.id ? 'primary.main' : 'divider',
              backgroundColor: selected === layout.id ? 'action.selected' : 'background.paper',
              transition: 'all 0.3s ease',
              '&:hover': disabled ? {} : {
                borderColor: 'primary.main',
                transform: 'translateY(-4px)',
                boxShadow: 4
              },
            }}
            onClick={() => !disabled && onChange(layout.id)}
          >
            <CardContent
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 1,
                p: 3,
              }}
            >
              <FormControlLabel
                value={layout.id}
                control={<Radio />}
                label=""
                sx={{
                  margin: 0,
                  '& .MuiFormControlLabel-label': {
                    display: 'none'
                  }
                }}
              />
              <Box
                sx={{
                  color: selected === layout.id ? 'primary.main' : 'text.secondary'
                }}
              >
                {layout.icon}
              </Box>
              <Typography
                variant="h6"
                component="div"
                sx={{
                  fontWeight: 600,
                  color: selected === layout.id ? 'primary.main' : 'text.primary'
                }}
              >
                {layout.name}
              </Typography>
              <Typography
                variant="body2"
                color="text.secondary"
                align="center"
              >
                {layout.description}
              </Typography>
            </CardContent>
          </Card>
        ))}
      </RadioGroup>
    </FormControl>
  );
};
