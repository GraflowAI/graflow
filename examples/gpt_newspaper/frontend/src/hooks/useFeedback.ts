import { useCallback, useEffect, useRef, useState } from "react";
import { newspaperClient } from "../api/client";
import type { FeedbackResponsePayload, LogEvent } from "../api/types";

/** Request browser notification permission on first call; no-op afterwards. */
const requestNotificationPermission = (() => {
  let requested = false;
  return () => {
    if (requested || typeof Notification === "undefined") return;
    requested = true;
    if (Notification.permission === "default") {
      Notification.requestPermission();
    }
  };
})();

/** Show a browser notification for a HITL feedback request. */
const showFeedbackNotification = (entry: LogEvent) => {
  if (typeof Notification === "undefined" || Notification.permission !== "granted") return;
  const title = "Editorial Approval Required";
  const body = entry.prompt ?? entry.message ?? "A feedback request is waiting for your review.";
  const notification = new Notification(title, {
    body,
    tag: `hitl-${entry.feedbackId ?? "unknown"}`,
    requireInteraction: true,
  });
  // Focus the window when the notification is clicked
  notification.onclick = () => {
    window.focus();
    notification.close();
  };
};

export interface UseFeedbackResult {
  pendingFeedback: LogEvent | null;
  submitting: boolean;
  submitError: string | null;
  submitFeedback: (feedbackId: string, response: FeedbackResponsePayload) => Promise<void>;
}

export const useFeedback = (logEntries: LogEvent[]): UseFeedbackResult => {
  const [pendingFeedback, setPendingFeedback] = useState<LogEvent | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const processedRef = useRef(0);

  // Request notification permission early
  useEffect(() => {
    requestNotificationPermission();
  }, []);

  // Watch logEntries for feedback events
  useEffect(() => {
    const newEntries = logEntries.slice(processedRef.current);
    processedRef.current = logEntries.length;

    for (const entry of newEntries) {
      if (entry.type === "feedback_request") {
        setPendingFeedback(entry);
        setSubmitError(null);
        showFeedbackNotification(entry);
      } else if (entry.type === "feedback_resolved" || entry.type === "feedback_timeout") {
        // Only clear if the resolved/timed-out event matches the current pending feedback.
        // Without this check, resolving article A's feedback would wrongly clear
        // the panel that is showing article B's feedback request.
        setPendingFeedback((prev) => {
          if (!prev) return null;
          if (!entry.feedbackId || prev.feedbackId === entry.feedbackId) return null;
          return prev;
        });
      }
    }
  }, [logEntries]);

  const submitFeedback = useCallback(
    async (feedbackId: string, response: FeedbackResponsePayload) => {
      setSubmitting(true);
      setSubmitError(null);
      try {
        await newspaperClient.respondToFeedback(feedbackId, response);
        // Optimistically clear the panel on successful API response instead of
        // relying solely on the WebSocket feedback_resolved event, which may be
        // delayed or lost (e.g. WebSocket reconnect, rapid log messages).
        setPendingFeedback((prev) =>
          prev?.feedbackId === feedbackId ? null : prev
        );
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : "Failed to submit feedback";
        setSubmitError(message);
      } finally {
        setSubmitting(false);
      }
    },
    []
  );

  return { pendingFeedback, submitting, submitError, submitFeedback };
};
