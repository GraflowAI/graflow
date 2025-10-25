import type { Preview } from "@storybook/react";
import { CssBaseline, ThemeProvider } from "@mui/material";
import theme from "../src/theme";

const preview: Preview = {
  decorators: [
    (Story) => (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Story />
      </ThemeProvider>
    )
  ]
};

export default preview;
