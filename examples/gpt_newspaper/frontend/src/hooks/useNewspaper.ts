import { AxiosError } from "axios";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { GPTNewspaperClient, newspaperClient } from "../api/client";
import type {
  LayoutOption,
  LogEvent,
  NewspaperResponse,
  NewspaperSummary,
  WorkflowOption
} from "../api/types";

export interface GeneratePayload {
  queries: string[];
  layout: LayoutOption;
  outputDir?: string;
  runId?: string;
  workflow: WorkflowOption;
}

export interface UseNewspaperResult {
  history: NewspaperSummary[];
  latest: NewspaperResponse | null;
  loading: boolean;
  error: string | null;
  logEntries: LogEvent[];
  activeRunId: string | null;
  generate: (payload: GeneratePayload) => Promise<NewspaperResponse | null>;
  refreshHistory: () => Promise<void>;
}

export const useNewspaper = (client: GPTNewspaperClient = newspaperClient): UseNewspaperResult => {
  const [history, setHistory] = useState<NewspaperSummary[]>([]);
  const [latest, setLatest] = useState<NewspaperResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [logEntries, setLogEntries] = useState<LogEvent[]>([]);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const logSocketRef = useRef<WebSocket | null>(null);

  const safeClient = useMemo(() => client, [client]);

  const closeLogStream = useCallback(() => {
    if (logSocketRef.current) {
      logSocketRef.current.close();
      logSocketRef.current = null;
    }
  }, []);

  const createRunId = useCallback((): string => {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
      return crypto.randomUUID();
    }
    return `run_${Date.now()}_${Math.floor(Math.random() * 1_000_000)}`;
  }, []);

  const buildWebSocketUrl = useCallback(
    (runId: string) => {
      const baseUrl = safeClient.getBaseUrl();
      const url = new URL(baseUrl);
      url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
      url.pathname = `/ws/logs/${encodeURIComponent(runId)}`;
      url.search = "";
      url.hash = "";
      return url.toString();
    },
    [safeClient]
  );

  const openLogStream = useCallback(
    (runId: string) => {
      if (!runId) {
        return;
      }
      closeLogStream();
      const wsUrl = buildWebSocketUrl(runId);
      console.log(`[WebSocket] Connecting to ${wsUrl}`);
      const ws = new WebSocket(wsUrl);
      logSocketRef.current = ws;

      ws.onopen = () => {
        console.log(`[WebSocket] Connected to ${wsUrl}`);
      };

      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data) as LogEvent;
          console.log(`[WebSocket] Received:`, payload);
          setLogEntries((prev) => [...prev, payload]);
          if (payload.type === "complete") {
            ws.close();
          }
        } catch (error) {
          console.error(`[WebSocket] Failed to parse message:`, error);
        }
      };

      ws.onerror = (error) => {
        console.error(`[WebSocket] Error:`, error);
        ws.close();
      };

      ws.onclose = (event) => {
        console.log(`[WebSocket] Closed (code: ${event.code}, reason: ${event.reason})`);
        if (logSocketRef.current === ws) {
          logSocketRef.current = null;
        }
      };
    },
    [buildWebSocketUrl, closeLogStream]
  );

  const refreshHistory = useCallback(async () => {
    try {
      const items = await safeClient.listNewspapers();
      setHistory(items);
    } catch (err: unknown) {
      const message =
        err instanceof AxiosError
          ? err.response?.data?.detail ?? err.message
          : err instanceof Error
            ? err.message
            : "Failed to load history.";
      setError(message);
    }
  }, [safeClient]);

  const generate = useCallback(
    async (payload: GeneratePayload) => {
      setLoading(true);
      setError(null);
      const runId = payload.runId ?? createRunId();
      setLogEntries([]);
      setActiveRunId(runId);
      openLogStream(runId);
      try {
        const response = await safeClient.generateNewspaper({ ...payload, runId });
        setLatest(response);
        await refreshHistory();
        return response;
      } catch (err: unknown) {
        const message =
          err instanceof AxiosError
            ? err.response?.data?.detail ?? err.message
            : err instanceof Error
              ? err.message
              : "Failed to generate newspaper.";
        setError(message);
        if (err instanceof AxiosError && !err.response) {
          closeLogStream();
        }
        return null;
      } finally {
        setLoading(false);
      }
    },
    [closeLogStream, createRunId, openLogStream, refreshHistory, safeClient]
  );

  useEffect(() => {
    void refreshHistory();
  }, [refreshHistory]);

  useEffect(() => () => closeLogStream(), [closeLogStream]);

  return {
    history,
    latest,
    loading,
    error,
    logEntries,
    activeRunId,
    generate,
    refreshHistory
  };
};
