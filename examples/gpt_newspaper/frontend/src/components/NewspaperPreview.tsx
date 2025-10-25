import { Box, Link, Paper, Stack, Typography } from "@mui/material";

export interface NewspaperPreviewProps {
  html: string;
  outputPath: string;
  createdAt: string;
  queries: string[];
  layout: string;
}

const getBackendUrl = (path: string): string => {
  const baseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
  // Remove leading slash from path if present to avoid double slashes
  const cleanPath = path.startsWith("/") ? path.slice(1) : path;
  return `${baseUrl}/${cleanPath}`;
};

const NewspaperPreview = ({ html, outputPath, createdAt, queries, layout }: NewspaperPreviewProps) => (
  <Paper elevation={3} sx={{ padding: 3 }}>
    <Stack spacing={2}>
      <Box>
        <Typography variant="h6">Latest newspaper</Typography>
        <Typography variant="body2" color="text.secondary">
          Generated at {new Date(createdAt).toLocaleString()} using the {layout} layout for: {queries.join(", ")}
        </Typography>
      </Box>
      <Link href={getBackendUrl(outputPath)} target="_blank" rel="noopener noreferrer">
        Open HTML file
      </Link>
      <Box
        sx={{
          border: (theme) => `1px solid ${theme.palette.divider}`,
          borderRadius: 2,
          overflow: "hidden",
          backgroundColor: "background.paper"
        }}
      >
        <iframe
          title="Generated newspaper preview"
          srcDoc={html}
          style={{ width: "100%", height: "600px", border: "none" }}
        />
      </Box>
    </Stack>
  </Paper>
);

export default NewspaperPreview;
