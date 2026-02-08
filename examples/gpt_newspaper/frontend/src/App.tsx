import { useCallback, useState } from "react";
import {
  Alert,
  CircularProgress,
  Container,
  Divider,
  List,
  ListItemButton,
  ListItemText,
  Paper,
  Stack,
  Typography
} from "@mui/material";
import Grid from "@mui/material/Grid";
import QueryForm from "./components/QueryForm";
import NewspaperPreview from "./components/NewspaperPreview";
import type { NewspaperPreviewProps } from "./components/NewspaperPreview";
import LogConsole from "./components/LogConsole";
import FeedbackPanel from "./components/FeedbackPanel";
import { useNewspaper } from "./hooks/useNewspaper";
import { useFeedback } from "./hooks/useFeedback";
import type { NewspaperSummary } from "./api/types";

const getBackendUrl = (path: string): string => {
  const baseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
  const cleanPath = path.startsWith("/") ? path.slice(1) : path;
  return `${baseUrl}/${cleanPath}`;
};

const App = () => {
  const { history, latest, loading, error, logEntries, activeRunId, generate } = useNewspaper();
  const { pendingFeedback, submitting, submitError, submitFeedback } = useFeedback(logEntries);
  const [selectedPreview, setSelectedPreview] = useState<NewspaperPreviewProps | null>(null);
  const [fetchingPreview, setFetchingPreview] = useState(false);

  const handleSubmit = useCallback(
    async (payload: Parameters<typeof generate>[0]) => {
      setSelectedPreview(null);
      await generate(payload);
    },
    [generate]
  );

  const handleSelectHistory = useCallback(
    async (item: NewspaperSummary) => {
      setFetchingPreview(true);
      try {
        const url = getBackendUrl(item.outputPath);
        const res = await fetch(url);
        if (!res.ok) throw new Error(`Failed to fetch: ${res.status}`);
        const html = await res.text();
        setSelectedPreview({
          html,
          outputPath: item.outputPath,
          createdAt: item.createdAt,
        });
      } catch (err) {
        console.error("Failed to load past newspaper", err);
      } finally {
        setFetchingPreview(false);
      }
    },
    []
  );

  // Show selected past run or latest generation result
  const preview = selectedPreview ?? (latest
    ? {
        html: latest.html,
        outputPath: latest.outputPath,
        createdAt: latest.createdAt,
        queries: latest.queries,
        layout: latest.layout,
        workflow: latest.workflow,
      }
    : null);

  return (
    <Container maxWidth="lg" sx={{ py: 6 }}>
      <Stack spacing={4}>
        <Stack spacing={1}>
          <Typography variant="h3">GPT Newspaper</Typography>
          <Typography color="text.secondary">
            Compose personalised newspapers by orchestrating the Graflow GPT workflow through a FastAPI backend.
          </Typography>
        </Stack>

        <Grid container spacing={4}>
          <Grid item xs={12} md={5}>
            <Paper elevation={1} sx={{ p: 3 }}>
              <Stack spacing={3}>
                <Typography variant="h5">Generate a newspaper</Typography>
                <Typography color="text.secondary">
                  Provide one or more topics. The backend runs the Graflow workflow and produces a fully rendered HTML
                  file.
                </Typography>
                <QueryForm onSubmit={handleSubmit} disabled={loading} />
                {loading && (
                  <Stack direction="row" spacing={1} alignItems="center">
                    <CircularProgress size={20} />
                    <Typography variant="body2">Generating newspaper…</Typography>
                  </Stack>
                )}
                {error && <Alert severity="error">{error}</Alert>}
              </Stack>
            </Paper>
            <Paper elevation={0} sx={{ p: 3, mt: 3 }}>
              <Typography variant="h6" gutterBottom>
                Recent outputs
              </Typography>
              {history.length === 0 ? (
                <Typography color="text.secondary">No newspapers generated yet.</Typography>
              ) : (
                <List dense>
                  {history.map((item) => (
                    <ListItemButton
                      key={item.outputPath}
                      selected={preview?.outputPath === item.outputPath}
                      onClick={() => handleSelectHistory(item)}
                      disabled={fetchingPreview}
                    >
                      <ListItemText
                        primary={item.filename}
                        secondary={new Date(item.createdAt).toLocaleString()}
                      />
                    </ListItemButton>
                  ))}
                </List>
              )}
            </Paper>
            {(activeRunId || logEntries.length > 0) && (
              <LogConsole entries={logEntries} runId={activeRunId} loading={loading} />
            )}
            {pendingFeedback && (
              <FeedbackPanel
                feedback={pendingFeedback}
                onSubmit={submitFeedback}
                submitting={submitting}
                error={submitError}
              />
            )}
          </Grid>
          <Grid item xs={12} md={7}>
            {fetchingPreview ? (
              <Paper variant="outlined" sx={{ p: 4, height: "100%" }}>
                <Stack spacing={2} alignItems="center" justifyContent="center" sx={{ height: "100%" }}>
                  <CircularProgress size={32} />
                  <Typography color="text.secondary">Loading newspaper…</Typography>
                </Stack>
              </Paper>
            ) : preview ? (
              <NewspaperPreview {...preview} />
            ) : (
              <Paper variant="outlined" sx={{ p: 4, height: "100%" }}>
                <Stack spacing={2} alignItems="center" justifyContent="center" sx={{ height: "100%" }}>
                  <Typography variant="h5">Preview</Typography>
                  <Typography color="text.secondary" align="center">
                    Generate a newspaper or select a past run to see the HTML preview here.
                  </Typography>
                </Stack>
              </Paper>
            )}
          </Grid>
        </Grid>

        <Divider />
        <Typography variant="body2" color="text.secondary">
          Backend API base URL: {import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000"}
        </Typography>
      </Stack>
    </Container>
  );
};

export default App;
