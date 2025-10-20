import type { Preview } from "@storybook/react";
import { ThemeProvider, CssBaseline } from '@mui/material';
import { theme } from '../src/theme';
import React from 'react';

const preview: Preview = {
  parameters: {
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/i,
      },
    },
    a11y: {
      config: {
        rules: [
          {
            // Disable color contrast checks for demo purposes
            id: 'color-contrast',
            enabled: false,
          },
        ],
      },
    },
  },
  decorators: [
    (Story) => (
      React.createElement(ThemeProvider, { theme }, [
        React.createElement(CssBaseline, { key: 'css' }),
        React.createElement(Story, { key: 'story' })
      ])
    ),
  ],
};

export default preview;
