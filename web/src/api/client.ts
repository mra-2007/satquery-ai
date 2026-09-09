// Thin fetch wrappers over the real FastAPI backend. No mocked data, no
// fabricated fallback values -- a failed request is surfaced as an error,
// never silently replaced with something invented.

import type { ChangeSummary, LegendEntry, QueryResponse, SceneSummary, UploadResponse } from "./types";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8080";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, init);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      // response body wasn't JSON -- fall back to statusText
    }
    throw new ApiError(response.status, detail);
  }
  return response.json() as Promise<T>;
}

export function getScenes(): Promise<SceneSummary[]> {
  return request<SceneSummary[]>("/scenes");
}

export function getSceneLegend(sceneId: string): Promise<LegendEntry[]> {
  return request<LegendEntry[]>(`/scenes/${encodeURIComponent(sceneId)}/legend`);
}

export function sceneImageUrl(sceneId: string, when?: "before" | "after"): string {
  const query = when ? `?when=${when}` : "";
  return `${API_URL}/scenes/${encodeURIComponent(sceneId)}/image${query}`;
}

export function sceneMaskUrl(sceneId: string, when?: "before" | "after"): string {
  const query = when ? `?when=${when}` : "";
  return `${API_URL}/scenes/${encodeURIComponent(sceneId)}/mask${query}`;
}

export function sceneChangeMaskUrl(sceneId: string): string {
  return `${API_URL}/scenes/${encodeURIComponent(sceneId)}/change-mask`;
}

export function getSceneChangeSummary(sceneId: string): Promise<ChangeSummary> {
  return request<ChangeSummary>(`/scenes/${encodeURIComponent(sceneId)}/change`);
}

export function postQuery(sceneId: string, query: string): Promise<QueryResponse> {
  return request<QueryResponse>("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scene_id: sceneId, query }),
  });
}

export function reportUrl(reportId: string): string {
  return `${API_URL}/report?report_id=${encodeURIComponent(reportId)}`;
}

/**
 * POST /upload always answers 200 with {ok: false, reason} for a rejected
 * image -- a bad upload is an expected outcome, not a transport error --
 * so this never throws for that case, only for a real network/server failure.
 */
export function uploadScene(formData: FormData): Promise<UploadResponse> {
  return request<UploadResponse>("/upload", { method: "POST", body: formData });
}
