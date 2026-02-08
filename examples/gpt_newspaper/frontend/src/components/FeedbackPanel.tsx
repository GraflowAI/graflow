import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Collapse,
  Divider,
  Link,
  Paper,
  Stack,
  TextField,
  Typography
} from "@mui/material";
import type { FeedbackResponsePayload, LogEvent } from "../api/types";

export interface FeedbackPanelProps {
  feedback: LogEvent;
  onSubmit: (feedbackId: string, response: FeedbackResponsePayload) => Promise<void>;
  submitting: boolean;
  error: string | null;
}

/** Extract article metadata from the feedback event. */
const useArticleMeta = (feedback: LogEvent) =>
  useMemo(() => {
    const m = feedback.metadata ?? {};
    console.log("[FeedbackPanel] metadata:", JSON.stringify(m, null, 2)?.slice(0, 1000));

    // Try body (pre-joined string) -> paragraphs (list) -> summary
    let body = typeof m.body === "string" && m.body ? m.body : "";
    if (!body && Array.isArray(m.paragraphs) && m.paragraphs.length > 0) {
      body = (m.paragraphs as string[]).join("\n\n");
    }
    if (!body && typeof m.summary === "string" && m.summary) {
      body = m.summary;
    }
    return {
      title: (m.title as string) ?? "",
      body,
      sources: (m.sources as Array<Record<string, unknown>>) ?? [],
      image: (m.image as string) ?? "",
      query: (m.query as string) ?? "",
    };
  }, [feedback.metadata]);

const FeedbackPanel = ({ feedback, onSubmit, submitting, error }: FeedbackPanelProps) => {
  const [reason, setReason] = useState("");
  const [showRejectReason, setShowRejectReason] = useState(false);
  const [remainingSeconds, setRemainingSeconds] = useState<number | null>(null);
  const [articleExpanded, setArticleExpanded] = useState(true);

  const feedbackId = feedback.feedbackId ?? "";
  const timeoutMs = (feedback.timeout ?? 300) * 1000;
  const createdAt = useMemo(() => new Date(feedback.timestamp).getTime(), [feedback.timestamp]);
  const article = useArticleMeta(feedback);

  // Countdown timer
  useEffect(() => {
    const update = () => {
      const elapsed = Date.now() - createdAt;
      const remaining = Math.max(0, Math.ceil((timeoutMs - elapsed) / 1000));
      setRemainingSeconds(remaining);
    };
    update();
    const interval = setInterval(update, 1000);
    return () => clearInterval(interval);
  }, [createdAt, timeoutMs]);

  const handleApprove = useCallback(async () => {
    await onSubmit(feedbackId, { approved: true });
  }, [feedbackId, onSubmit]);

  const handleReject = useCallback(async () => {
    if (!showRejectReason) {
      setShowRejectReason(true);
      return;
    }
    await onSubmit(feedbackId, { approved: false, reason: reason || undefined });
  }, [feedbackId, onSubmit, reason, showRejectReason]);

  const formatTime = (seconds: number): string => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  const hasArticleContent = Boolean(article.body || article.title);

  return (
    <Paper elevation={3} sx={{ p: 3, mt: 3, border: 2, borderColor: "warning.main" }}>
      <Stack spacing={2}>
        <Alert severity="warning" sx={{ fontWeight: "bold" }}>
          Editorial Approval Required
        </Alert>

        {/* ---- Article preview ---- */}
        {hasArticleContent ? (
          <>
            <Stack direction="row" alignItems="center" justifyContent="space-between">
              <Typography variant="h6">{article.title || "Untitled"}</Typography>
              <Link
                component="button"
                type="button"
                variant="body2"
                underline="hover"
                onClick={() => setArticleExpanded((prev) => !prev)}
              >
                {articleExpanded ? "Collapse" : "Expand"} article
              </Link>
            </Stack>

            {article.query && (
              <Chip label={article.query} size="small" color="primary" variant="outlined" />
            )}

            <Collapse in={articleExpanded}>
              <Stack spacing={2}>
                {article.image && (
                  <Box
                    component="img"
                    src={article.image}
                    alt={article.title}
                    sx={{
                      width: "100%",
                      maxHeight: 200,
                      objectFit: "cover",
                      borderRadius: 1,
                    }}
                  />
                )}

                <Paper
                  variant="outlined"
                  sx={{
                    p: 2,
                    maxHeight: 400,
                    overflow: "auto",
                    bgcolor: "grey.50",
                  }}
                >
                  <Typography
                    variant="body2"
                    sx={{ whiteSpace: "pre-wrap", lineHeight: 1.7 }}
                  >
                    {article.body}
                  </Typography>
                </Paper>

                {article.sources.length > 0 && (
                  <Box>
                    <Typography variant="subtitle2" gutterBottom>
                      Sources ({article.sources.length})
                    </Typography>
                    <Stack spacing={0.5}>
                      {article.sources.map((src, idx) => {
                        const url = (src.url ?? src.href ?? "") as string;
                        const name = (src.name ?? src.title ?? url) as string;
                        return (
                          <Typography key={idx} variant="caption" color="text.secondary">
                            {url ? (
                              <Link href={url} target="_blank" rel="noopener noreferrer" underline="hover">
                                {name || url}
                              </Link>
                            ) : (
                              name
                            )}
                          </Typography>
                        );
                      })}
                    </Stack>
                  </Box>
                )}
              </Stack>
            </Collapse>

            <Divider />
          </>
        ) : (
          /* Fallback: show raw prompt when no structured metadata is available */
          <Typography variant="body1" sx={{ whiteSpace: "pre-wrap" }}>
            {feedback.prompt}
          </Typography>
        )}

        {remainingSeconds !== null && (
          <Typography variant="body2" color="text.secondary">
            Time remaining: {formatTime(remainingSeconds)}
          </Typography>
        )}

        {showRejectReason && (
          <TextField
            label="Reason for rejection (optional)"
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            multiline
            rows={2}
            disabled={submitting}
            size="small"
          />
        )}

        {error && <Alert severity="error">{error}</Alert>}

        <Stack direction="row" spacing={2}>
          <Button
            variant="contained"
            color="success"
            onClick={handleApprove}
            disabled={submitting}
            startIcon={submitting ? <CircularProgress size={16} /> : undefined}
          >
            Approve
          </Button>
          <Button
            variant="contained"
            color="error"
            onClick={handleReject}
            disabled={submitting}
            startIcon={submitting ? <CircularProgress size={16} /> : undefined}
          >
            {showRejectReason ? "Submit Rejection" : "Reject"}
          </Button>
          {showRejectReason && (
            <Button
              variant="text"
              size="small"
              onClick={() => setShowRejectReason(false)}
              disabled={submitting}
            >
              Cancel
            </Button>
          )}
        </Stack>
      </Stack>
    </Paper>
  );
};

export default FeedbackPanel;
