import type { Meta, StoryObj } from "@storybook/react";
import NewspaperPreview from "./NewspaperPreview";

const sampleHtml = `
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>Sample Newspaper</title>
  </head>
  <body>
    <article>
      <h2>Breaking News</h2>
      <p>This is a preview of the generated newspaper content.</p>
    </article>
  </body>
</html>
`;

const meta: Meta<typeof NewspaperPreview> = {
  title: "Components/NewspaperPreview",
  component: NewspaperPreview,
  args: {
    createdAt: new Date().toISOString(),
    outputPath: "/tmp/newspaper.html",
    queries: ["sample", "news"],
    html: sampleHtml
  }
};

export default meta;

type Story = StoryObj<typeof NewspaperPreview>;

export const Default: Story = {};
