// Thin fetch wrappers over the real FastAPI backend. No mocked data, no
// fabricated fallback values -- a failed request is surfaced as an error,
// never silently replaced with something invented.

import type {
  ChangeSummary,
  CrossModalSensor,
  CrossModalSummary,
  LegendEntry,
  QueryResponse,
  SceneSummary,
  UploadResponse,
} from "./types";

// Same origin by default (empty string -- `${API_URL}${path}` then resolves
// relative to whatever page served this bundle), so a production build
// works unmodified wherever it's deployed: api/main.py serves this build's
// static files itself, on the SAME port as the API, once web/dist exists
// (see its own module docstring). Local dev (`npm run dev`, import.meta.env.DEV
// -- a real Vite build-time flag, not manually set) keeps defaulting to the
// separately-running backend on :8080, exactly as before. VITE_API_URL, if
// set, always wins over either default.
const API_URL = import.meta.env.VITE_API_URL ?? (import.meta.env.DEV ? "http://localhost:8080" : "");

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

export function sceneImageUrl(sceneId: string, when?: "before" | "after", cloudSimulation?: boolean): string {
  const params = new URLSearchParams();
  if (when) params.set("when", when);
  if (cloudSimulation) params.set("cloud_simulation", "true");
  const query = params.toString();
  return `${API_URL}/scenes/${encodeURIComponent(sceneId)}/image${query ? `?${query}` : ""}`;
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

export function sceneCrossModalMaskUrl(
  sceneId: string,
  sensor: CrossModalSensor,
  cloudSimulation: boolean,
): string {
  const params = new URLSearchParams({ sensor });
  if (cloudSimulation) params.set("cloud_simulation", "true");
  return `${API_URL}/scenes/${encodeURIComponent(sceneId)}/cross-modal-mask?${params}`;
}

export function getSceneCrossModal(sceneId: string, cloudSimulation: boolean): Promise<CrossModalSummary> {
  const params = new URLSearchParams();
  if (cloudSimulation) params.set("cloud_simulation", "true");
  const query = params.toString();
  return request<CrossModalSummary>(`/scenes/${encodeURIComponent(sceneId)}/cross-modal${query ? `?${query}` : ""}`);
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
