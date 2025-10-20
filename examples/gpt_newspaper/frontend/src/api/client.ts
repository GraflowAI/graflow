/**
 * Type-safe Axios API client for GPT Newspaper backend
 */

import axios, { AxiosInstance, AxiosError, AxiosRequestConfig } from 'axios';
import type {
  NewspaperRequest,
  NewspaperResponse,
  HealthResponse,
  APIError
} from '../types';

/**
 * API client configuration
 */
export interface APIClientConfig {
  baseURL: string;
  timeout?: number;
}

/**
 * Custom error class for API errors
 */
export class APIClientError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public details?: string
  ) {
    super(message);
    this.name = 'APIClientError';
  }
}

/**
 * Type-safe API client using Axios
 */
export class APIClient {
  private client: AxiosInstance;

  constructor(config: APIClientConfig) {
    this.client = axios.create({
      baseURL: config.baseURL,
      timeout: config.timeout || 300000, // 5 minutes default
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor for logging
    this.client.interceptors.request.use(
      (config) => {
        console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('[API] Request error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => {
        console.log(`[API] Response ${response.status} from ${response.config.url}`);
        return response;
      },
      (error: AxiosError<APIError>) => {
        return this.handleError(error);
      }
    );
  }

  /**
   * Handle API errors and convert to custom error format
   */
  private handleError(error: AxiosError<APIError>): Promise<never> {
    if (error.response) {
      // Server responded with error
      const message = error.response.data?.detail || error.message;
      throw new APIClientError(
        message,
        error.response.status,
        error.response.data?.detail
      );
    } else if (error.request) {
      // Request made but no response
      throw new APIClientError(
        'No response from server. Please check if the backend is running.',
        undefined,
        'Network error'
      );
    } else {
      // Error setting up request
      throw new APIClientError(
        error.message,
        undefined,
        'Request setup error'
      );
    }
  }

  /**
   * Check API health status
   */
  async checkHealth(): Promise<HealthResponse> {
    const response = await this.client.get<HealthResponse>('/');
    return response.data;
  }

  /**
   * Generate newspaper from topics
   */
  async generateNewspaper(
    request: NewspaperRequest,
    config?: AxiosRequestConfig
  ): Promise<NewspaperResponse> {
    const response = await this.client.post<NewspaperResponse>(
      '/api/generate',
      request,
      config
    );
    return response.data;
  }

  /**
   * Get the base URL
   */
  getBaseURL(): string {
    return this.client.defaults.baseURL || '';
  }
}

/**
 * Create and export default API client instance
 */
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const apiClient = new APIClient({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes
});

/**
 * Export factory function for testing
 */
export const createAPIClient = (config: APIClientConfig): APIClient => {
  return new APIClient(config);
};
