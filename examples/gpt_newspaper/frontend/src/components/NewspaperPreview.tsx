import { useMemo } from "react";
import { Box, Link, Paper, Stack, Typography } from "@mui/material";

export interface NewspaperPreviewProps {
  html: string;
  outputPath: string;
  createdAt: string;
  queries?: string[];
  layout?: string;
  workflow?: string;
}

const getBackendUrl = (path: string): string => {
  const baseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
  const cleanPath = path.startsWith("/") ? path.slice(1) : path;
  return `${baseUrl}/${cleanPath}`;
};

const NewspaperPreview = ({ html, outputPath, createdAt, queries, layout, workflow }: NewspaperPreviewProps) => {
  // Inject <base> tag so relative links (e.g. article HTML files) resolve
  // against the backend directory instead of the frontend origin.
  const htmlWithBase = useMemo(() => {
    const fileUrl = getBackendUrl(outputPath);
    const baseHref = fileUrl.substring(0, fileUrl.lastIndexOf("/") + 1);
    if (html.includes("<head>")) {
      return html.replace("<head>", `<head><base href="${baseHref}" target="_blank">`);
    }
    return `<base href="${baseHref}" target="_blank">${html}`;
  }, [html, outputPath]);
  const details: string[] = [];
  if (layout) details.push(`${layout} layout`);
  if (workflow) details.push(`${workflow} workflow`);
  if (queries?.length) details.push(`topics: ${queries.join(", ")}`);

  return (
    <Paper elevation={3} sx={{ padding: 3 }}>
      <Stack spacing={2}>
        <Box>
          <Typography variant="h6">Newspaper preview</Typography>
          <Typography variant="body2" color="text.secondary">
            Generated at {new Date(createdAt).toLocaleString()}
            {details.length > 0 && ` — ${details.join(" / ")}`}
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
            srcDoc={htmlWithBase}
            style={{ width: "100%", height: "600px", border: "none" }}
          />
        </Box>
      </Stack>
    </Paper>
  );
};

export default NewspaperPreview;
