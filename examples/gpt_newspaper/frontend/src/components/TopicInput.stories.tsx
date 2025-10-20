/**
 * Storybook stories for TopicInput component
 */

import type { Meta, StoryObj } from '@storybook/react';
import { fn } from '@storybook/test';
import { TopicInput } from './TopicInput';

const meta = {
  title: 'Components/TopicInput',
  component: TopicInput,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
  argTypes: {
    id: {
      control: 'number',
      description: 'Unique identifier for the topic',
    },
    value: {
      control: 'text',
      description: 'Current value of the input',
    },
    isFirst: {
      control: 'boolean',
      description: 'Whether this is the first topic (cannot be removed)',
    },
    isLast: {
      control: 'boolean',
      description: 'Whether this is the last topic (shows add/remove buttons)',
    },
    canAddMore: {
      control: 'boolean',
      description: 'Whether more topics can be added',
    },
    disabled: {
      control: 'boolean',
      description: 'Whether the input is disabled',
    },
  },
  args: {
    onChange: fn(),
    onAdd: fn(),
    onRemove: fn(),
  },
} satisfies Meta<typeof TopicInput>;

export default meta;
type Story = StoryObj<typeof meta>;

// Default story
export const Default: Story = {
  args: {
    id: 1,
    value: '',
    isFirst: true,
    isLast: true,
    canAddMore: true,
    disabled: false,
  },
};

// With value
export const WithValue: Story = {
  args: {
    id: 1,
    value: 'AI developments',
    isFirst: true,
    isLast: true,
    canAddMore: true,
    disabled: false,
  },
};

// Middle topic (not first, not last)
export const MiddleTopic: Story = {
  args: {
    id: 2,
    value: 'Climate updates',
    isFirst: false,
    isLast: false,
    canAddMore: true,
    disabled: false,
  },
};

// Last topic with remove button
export const LastTopicWithRemove: Story = {
  args: {
    id: 3,
    value: 'Technology trends',
    isFirst: false,
    isLast: true,
    canAddMore: true,
    disabled: false,
  },
};

// Maximum topics reached
export const MaximumReached: Story = {
  args: {
    id: 10,
    value: 'Final topic',
    isFirst: false,
    isLast: true,
    canAddMore: false,
    disabled: false,
  },
};

// Disabled state
export const Disabled: Story = {
  args: {
    id: 1,
    value: 'AI developments',
    isFirst: true,
    isLast: true,
    canAddMore: true,
    disabled: true,
  },
};
