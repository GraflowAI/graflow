import { Box, Chip, IconButton, Paper, Stack, Tooltip, Typography } from "@mui/material";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";
import type { LogEvent } from "../api/types";

export interface LogConsoleProps {
  entries: LogEvent[];
  runId: string | null;
  loading: boolean;
}

const formatLine = (entry: LogEvent) => {
  const prefix = `[${new Date(entry.timestamp).toLocaleTimeString()}]`;
  if (entry.type === "feedback_request") {
    return `${prefix} [HITL] Waiting for editorial approval...`;
  }
  if (entry.type === "feedback_resolved") {
    return `${prefix} [HITL] Feedback received, resuming workflow`;
  }
  if (entry.type === "feedback_timeout") {
    return `${prefix} [HITL] Feedback timed out`;
  }
  const statusLabel = entry.type === "status" ? ` (${entry.status ?? "status"})` : "";
  const message = (entry.message ?? "").trimEnd();
  return `${prefix}${statusLabel} ${message}`.trim();
};

const openLogsInNewWindow = (entries: LogEvent[], runId: string | null) => {
  const initialLogsContent = entries.length === 0
    ? "Waiting for backend output…"
    : entries.map((entry) => formatLine(entry)).join("\n");

  const baseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
  const wsBaseUrl = baseUrl.replace(/^http/, "ws");
  const wsUrl = runId ? `${wsBaseUrl}/ws/logs/${encodeURIComponent(runId)}` : "";

  const htmlContent = `
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8">
        <title>Live Logs - ${runId || "Unknown Run"}</title>
        <style>
          body {
            margin: 0;
            padding: 20px;
            font-family: 'Roboto Mono', 'Menlo', monospace;
            background-color: #1a1a1a;
            color: #f5f5f5;
          }
          .header {
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #444;
          }
          h1 {
            margin: 0 0 10px 0;
            font-size: 20px;
            font-weight: 600;
          }
          .run-id {
            font-size: 12px;
            color: #888;
          }
          .status {
            font-size: 12px;
            color: #4caf50;
            margin-top: 5px;
          }
          .status.disconnected {
            color: #f44336;
          }
          pre {
            margin: 0;
            white-space: pre-wrap;
            word-break: break-word;
            font-size: 13px;
            line-height: 1.5;
          }
          .auto-scroll {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 8px 16px;
            background: #333;
            color: #fff;
            border: 1px solid #555;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
          }
          .auto-scroll:hover {
            background: #444;
          }
        </style>
      </head>
      <body>
        <div class="header">
          <h1>Live Logs</h1>
          ${runId ? `<div class="run-id">Run ID: ${runId}</div>` : ""}
          <div class="status" id="status">Connecting...</div>
        </div>
        <pre id="logs">${initialLogsContent}</pre>
        <button class="auto-scroll" onclick="toggleAutoScroll()" id="scrollBtn">Auto-scroll: ON</button>
        <script>
          let autoScroll = true;
          const logsEl = document.getElementById('logs');
          const scrollBtn = document.getElementById('scrollBtn');
          const statusEl = document.getElementById('status');
          let ws = null;

          function toggleAutoScroll() {
            autoScroll = !autoScroll;
            scrollBtn.textContent = 'Auto-scroll: ' + (autoScroll ? 'ON' : 'OFF');
            if (autoScroll) {
              window.scrollTo(0, document.body.scrollHeight);
            }
          }

          function formatLogEntry(entry) {
            const timestamp = new Date(entry.timestamp).toLocaleTimeString();
            const prefix = \`[\${timestamp}]\`;
            const statusLabel = entry.type === "status" ? \` (\${entry.status || "status"})\` : "";
            const message = (entry.message || "").trimEnd();
            return \`\${prefix}\${statusLabel} \${message}\`.trim();
          }

          function scrollToBottom() {
            if (autoScroll) {
              window.scrollTo(0, document.body.scrollHeight);
            }
          }

          // Connect to WebSocket for live updates
          if ("${wsUrl}") {
            console.log("Connecting to WebSocket:", "${wsUrl}");
            ws = new WebSocket("${wsUrl}");

            ws.onopen = () => {
              console.log("WebSocket connected");
              statusEl.textContent = "Connected (Live)";
              statusEl.className = "status";
            };

            ws.onmessage = (event) => {
              try {
                const payload = JSON.parse(event.data);
                console.log("Received log:", payload);
                const logLine = formatLogEntry(payload);
                if (logsEl.textContent === "Waiting for backend output…") {
                  logsEl.textContent = logLine;
                } else {
                  logsEl.textContent += "\\n" + logLine;
                }
                scrollToBottom();

                if (payload.type === "complete") {
                  statusEl.textContent = "Workflow Complete";
                  ws.close();
                }
              } catch (error) {
                console.error("Failed to parse log message:", error);
              }
            };

            ws.onerror = (error) => {
              console.error("WebSocket error:", error);
              statusEl.textContent = "Connection Error";
              statusEl.className = "status disconnected";
            };

            ws.onclose = () => {
              console.log("WebSocket closed");
              if (statusEl.textContent !== "Workflow Complete") {
                statusEl.textContent = "Disconnected";
                statusEl.className = "status disconnected";
              }
            };

            // Close WebSocket when window closes
            window.addEventListener('beforeunload', () => {
              if (ws) {
                ws.close();
              }
            });
          } else {
            statusEl.textContent = "No active run";
            statusEl.className = "status disconnected";
          }

          // Initial scroll
          scrollToBottom();
        </script>
      </body>
    </html>
  `;

  const blob = new Blob([htmlContent], { type: "text/html" });
  const url = URL.createObjectURL(blob);
  window.open(url, "_blank", "width=800,height=600");
  // Release the object URL after the new window has loaded the content
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};

const LogConsole = ({ entries, runId, loading }: LogConsoleProps) => (
  <Paper variant="outlined" sx={{ p: 2, mt: 3 }}>
    <Stack spacing={1}>
      <Stack direction="row" spacing={1} alignItems="center">
        <Typography variant="h6">Live logs</Typography>
        {runId && <Chip size="small" label={runId} />}
        {loading && (
          <Typography variant="body2" color="text.secondary">
            Streaming…
          </Typography>
        )}
        <Tooltip title="Open logs in new window">
          <IconButton
            size="small"
            onClick={() => openLogsInNewWindow(entries, runId)}
            sx={{ ml: "auto" }}
          >
            <OpenInNewIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Stack>
      <Box
        component="pre"
        sx={{
          borderRadius: 1,
          bgcolor: "grey.900",
          color: "grey.50",
          p: 2,
          minHeight: 200,
          maxHeight: 320,
          overflowY: "auto",
          fontFamily: "Roboto Mono,Menlo,monospace",
          fontSize: "0.85rem",
          whiteSpace: "pre-wrap",
          wordBreak: "break-word"
        }}
      >
        {entries.length === 0 ? "Waiting for backend output…" : entries.map((entry) => formatLine(entry)).join("\n")}
      </Box>
    </Stack>
  </Paper>
);

export default LogConsole;
