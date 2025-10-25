import axios, { AxiosInstance } from "axios";
import type { LayoutOption, NewspaperRequest, NewspaperResponse, NewspaperSummary } from "./types";

const DEFAULT_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export class GPTNewspaperClient {
  private readonly http: AxiosInstance;
  private readonly baseURL: string;

  public constructor(baseURL: string = DEFAULT_BASE_URL) {
    this.baseURL = baseURL;
    this.http = axios.create({
      baseURL,
      headers: {
        "Content-Type": "application/json"
      }
    });
  }

  public async generateNewspaper(payload: NewspaperRequest): Promise<NewspaperResponse> {
    const filteredQueries = payload.queries.map((query) => query.trim()).filter(Boolean);
    const response = await this.http.post<NewspaperResponse>("/api/newspaper", {
      ...payload,
      queries: filteredQueries
    });
    return response.data;
  }

  public async listNewspapers(limit = 10): Promise<NewspaperSummary[]> {
    const response = await this.http.get<NewspaperSummary[]>("/api/newspaper", {
      params: { limit }
    });
    return response.data;
  }

  public withBaseUrl(baseURL: string): GPTNewspaperClient {
    return new GPTNewspaperClient(baseURL);
  }

  public getBaseUrl(): string {
    return this.baseURL;
  }
}

export const supportedLayouts: LayoutOption[] = ["single", "two-column"];

export const newspaperClient = new GPTNewspaperClient();
