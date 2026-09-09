// Mirrors evidence/schema.py and api/schemas.py exactly. Nothing here is
// speculative -- every field is one the backend actually returns.

export type SceneKind = "single" | "cross_modal" | "change";

export interface SceneSummary {
  id: string;
  filename: string;
  width: number;
  height: number;
  gsd_metres: number;
  sensor: string;
  kind: SceneKind;
  source: "demo" | "upload";
  warnings: string[];
  // Free-text, caller-supplied acquisition dates for a "change"-kind scene.
  // Null whenever not supplied (e.g. the OSCD-seeded demo scene) -- never
  // fabricated, so the UI must show a plain "BEFORE"/"AFTER" label instead
  // of inventing a date when these are null.
  before_date: string | null;
  after_date: string | null;
}

export interface ClassAreaChange {
  class_id: number;
  class_name: string;
  gained_ha: number;
  lost_ha: number;
  net_ha: number;
}

export interface ChangeSummary {
  summary: string;
  changed_pixels: number;
  changed_fraction: number;
  classes: ClassAreaChange[];
}

export interface UploadResponse {
  ok: boolean;
  reason: string | null;
  modality: string;
  gsd_metres: number | null;
  band_count: number | null;
  width: number | null;
  height: number | null;
  auto_shifted: boolean;
  fallback_single_modality: boolean;
  scene_id: string | null;
  kind: SceneKind | null;
  warnings: string[];
}

export interface LegendEntry {
  class_id: number;
  class_name: string;
  color: string;
}

export interface TraceStep {
  task: string;
  tool: string;
  parameters: Record<string, unknown>;
  output: unknown;
  confidence: number;
}

export type ClaimType = "size" | "count" | "presence" | "adjacency";

export interface VerificationClaim {
  text: string;
  claim_type: ClaimType;
  tool: string;
  class_name: string | null;
  claimed_value: boolean | number | null;
  recomputed_value: boolean | number | null;
  passed: boolean;
  reason: string | null;
}

// Not a top-level API response -- this is the shape of a TraceStep's
// `output` whenever `step.tool === "verify"` (tools/verifier.py's
// VerificationResult, as summarized by agent/executor.py's _summarize()).
export interface VerificationSummary {
  original_answer: string;
  verified_answer: string;
  all_passed: boolean;
  claims: VerificationClaim[];
}

export interface SourceConfidence {
  source: string;
  confidence: number;
}

export interface GeoJsonFeature {
  type: "Feature";
  geometry: unknown;
  properties: Record<string, unknown> | null;
  id: string | number | null;
}

export interface GeoJsonFeatureCollection {
  type: "FeatureCollection";
  features: GeoJsonFeature[];
}

export interface Evidence {
  schema_version: string;
  value: boolean | number | string;
  units: string | null;
  geometry: GeoJsonFeatureCollection;
  confidence: SourceConfidence[];
  execution_trace: TraceStep[];
}

export interface QueryResponse {
  report_id: string;
  scene_id: string;
  query: string;
  evidence: Evidence;
  limitation: string | null;
}

export interface ApiErrorBody {
  detail: string;
}
