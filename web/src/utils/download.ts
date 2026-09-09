/**
 * Triggers a browser download of `data` as pretty-printed JSON, entirely
 * client-side -- no network round trip, since the caller already has the
 * data in memory (e.g. evidence.geometry, already part of a /query
 * response). Mirrors GET /report's own approach of re-serving exactly
 * what was already computed, never recomputing or inventing anything.
 */
export function downloadJson(filename: string, data: unknown): void {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
