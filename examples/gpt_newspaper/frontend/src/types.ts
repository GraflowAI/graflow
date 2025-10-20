/**
 * Type definitions for GPT Newspaper API
 */

export interface NewspaperRequest {
  topics: string[];
  layout: string;
  max_workers?: number | null;
}

export interface NewspaperResponse {
  path: string;
  article_count: number;
  timestamp: number;
}

export interface HealthResponse {
  status: string;
  message: string;
}

export interface APIError {
  detail: string;
}

export type LayoutOption = 'layout_1.html' | 'layout_2.html' | 'layout_3.html';
