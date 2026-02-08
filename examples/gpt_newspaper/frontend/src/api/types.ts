export type LayoutOption = "single" | "two-column";
export type WorkflowOption = "original" | "dynamic" | "agent";

export interface NewspaperRequest {
  queries: string[];
  layout: LayoutOption;
  outputDir?: string;
  runId?: string;
  workflow?: WorkflowOption;
  enableHitl?: boolean;
}

export interface NewspaperResponse {
  outputPath: string;
  html: string;
  createdAt: string;
  layout: LayoutOption;
  queries: string[];
  runId: string;
  workflow: WorkflowOption;
}

export interface NewspaperSummary {
  filename: string;
  createdAt: string;
  outputPath: string;
  runId: string;
}

export type LogEventType = "log" | "status" | "complete" | "feedback_request" | "feedback_resolved" | "feedback_timeout";

export type FeedbackType = "approval" | "text" | "selection" | "multi_selection" | "custom";

export interface LogEvent {
  type: LogEventType;
  message?: string;
  status?: string;
  runId: string;
  timestamp: string;
  // Feedback-specific fields (present when type is "feedback_request")
  feedbackId?: string;
  taskId?: string;
  feedbackType?: FeedbackType;
  prompt?: string;
  options?: string[];
  metadata?: Record<string, unknown>;
  timeout?: number;
}

export interface FeedbackResponsePayload {
  approved?: boolean;
  reason?: string;
  text?: string;
  selected?: string;
  responded_by?: string;
}
