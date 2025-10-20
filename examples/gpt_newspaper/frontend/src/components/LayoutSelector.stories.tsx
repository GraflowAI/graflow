/**
 * Storybook stories for LayoutSelector component
 */

import type { Meta, StoryObj } from '@storybook/react';
import { fn } from '@storybook/test';
import { LayoutSelector } from './LayoutSelector';

const meta = {
  title: 'Components/LayoutSelector',
  component: LayoutSelector,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
  argTypes: {
    selected: {
      control: 'radio',
      options: ['layout_1.html', 'layout_2.html', 'layout_3.html'],
      description: 'Currently selected layout',
    },
    disabled: {
      control: 'boolean',
      description: 'Whether the selector is disabled',
    },
  },
  args: {
    onChange: fn(),
  },
} satisfies Meta<typeof LayoutSelector>;

export default meta;
type Story = StoryObj<typeof meta>;

// Default story - Layout 1 selected
export const Layout1Selected: Story = {
  args: {
    selected: 'layout_1.html',
    disabled: false,
  },
};

// Layout 2 selected
export const Layout2Selected: Story = {
  args: {
    selected: 'layout_2.html',
    disabled: false,
  },
};

// Layout 3 selected
export const Layout3Selected: Story = {
  args: {
    selected: 'layout_3.html',
    disabled: false,
  },
};

// Disabled state
export const Disabled: Story = {
  args: {
    selected: 'layout_1.html',
    disabled: true,
  },
};
