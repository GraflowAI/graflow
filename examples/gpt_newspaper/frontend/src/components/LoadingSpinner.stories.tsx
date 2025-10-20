/**
 * Storybook stories for LoadingSpinner component
 */

import type { Meta, StoryObj } from '@storybook/react';
import { LoadingSpinner } from './LoadingSpinner';

const meta = {
  title: 'Components/LoadingSpinner',
  component: LoadingSpinner,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  argTypes: {
    loading: {
      control: 'boolean',
      description: 'Whether to show the loading spinner',
    },
  },
} satisfies Meta<typeof LoadingSpinner>;

export default meta;
type Story = StoryObj<typeof meta>;

// Default loading state
export const Loading: Story = {
  args: {
    loading: true,
  },
};

// Hidden (not loading)
export const NotLoading: Story = {
  args: {
    loading: false,
  },
};
