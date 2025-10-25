import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import QueryForm from "./QueryForm";

const meta: Meta<typeof QueryForm> = {
  title: "Components/QueryForm",
  component: QueryForm,
  args: {
    disabled: false,
    onSubmit: fn()
  }
};

export default meta;

type Story = StoryObj<typeof QueryForm>;

export const Default: Story = {};

export const Disabled: Story = {
  args: {
    disabled: true
  }
};
