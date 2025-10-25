export type LayoutOption = "single" | "two-column";
export type WorkflowOption = "original" | "dynamic";

export interface NewspaperRequest {
  queries: string[];
  layout: LayoutOption;
  outputDir?: string;
  runId?: string;
  workflow?: WorkflowOption;
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

export type LogEventType = "log" | "status" | "complete";

export interface LogEvent {
  type: LogEventType;
  message?: string;
  status?: string;
  runId: string;
  timestamp: string;
}
