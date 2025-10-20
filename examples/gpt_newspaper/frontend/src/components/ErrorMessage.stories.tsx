/**
 * Storybook stories for ErrorMessage component
 */

import type { Meta, StoryObj } from '@storybook/react';
import { fn } from '@storybook/test';
import { ErrorMessage } from './ErrorMessage';

const meta = {
  title: 'Components/ErrorMessage',
  component: ErrorMessage,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
  argTypes: {
    message: {
      control: 'text',
      description: 'Error message to display',
    },
    title: {
      control: 'text',
      description: 'Optional error title',
    },
    severity: {
      control: 'radio',
      options: ['error', 'warning', 'info'],
      description: 'Severity level',
    },
    open: {
      control: 'boolean',
      description: 'Whether to show the error',
    },
  },
  args: {
    onDismiss: fn(),
  },
} satisfies Meta<typeof ErrorMessage>;

export default meta;
type Story = StoryObj<typeof meta>;

// Default error
export const Error: Story = {
  args: {
    message: 'An error occurred while processing your request.',
    title: 'Error',
    severity: 'error',
    open: true,
  },
};

// Warning
export const Warning: Story = {
  args: {
    message: 'Please check your API configuration.',
    title: 'Warning',
    severity: 'warning',
    open: true,
  },
};

// Info
export const Info: Story = {
  args: {
    message: 'Processing may take a few minutes.',
    title: 'Information',
    severity: 'info',
    open: true,
  },
};

// No title
export const NoTitle: Story = {
  args: {
    message: 'Please fill in at least one topic.',
    severity: 'error',
    open: true,
  },
};

// Without dismiss button
export const NoDismiss: Story = {
  args: {
    message: 'This is a permanent error message.',
    title: 'Error',
    severity: 'error',
    open: true,
    onDismiss: undefined,
  },
};

// Long message
export const LongMessage: Story = {
  args: {
    message:
      'An unexpected error occurred while generating the newspaper. The API request timed out after 5 minutes. This could be due to heavy load on the server or network issues. Please try again later or contact support if the problem persists.',
    title: 'Error',
    severity: 'error',
    open: true,
  },
};

// Hidden
export const Hidden: Story = {
  args: {
    message: 'This message should not be visible.',
    title: 'Error',
    severity: 'error',
    open: false,
  },
};
