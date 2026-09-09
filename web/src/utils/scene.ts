/**
 * Sentinel-2 product filenames embed their real acquisition date/time
 * (ESA's standard naming convention, e.g.
 * "S2A_MSIL2A_20170617T113321_N9999_R080_T29UPU_13_55.npz" acquired
 * 2017-06-17). Reading it back out is not fabrication -- it's the same
 * genuine metadata the file has always carried, just not exposed by
 * GET /scenes as a separate field.
 */
export function parseAcquisitionDate(filename: string): string | null {
  const match = filename.match(/_(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})_/);
  if (!match) return null;
  const [, year, month, day] = match;
  return `${year}-${month}-${day}`;
}

export function parsePlatform(filename: string): string | null {
  const match = filename.match(/^(S2[AB])_/);
  return match ? match[1] : null;
}

// Preferred flagship demo scene. Six patches with real Urban fabric AND
// water were added to data/demo_patches/ after the original 9 (none of
// which had both at once), and this is the one patch, of all 15, whose
// PREDICTED mask (the real pipeline, not ground truth) answers all four
// ANALYZE example questions non-zero/true: 14 water bodies, 90.82 ha
// forest, built-up-near-water = yes, and a capability-guardrail-degraded
// building answer (0.42 ha, real Urban fabric present but too small at
// 10 m GSD to count -- the "exact answer withheld" state, the product's
// signature moment). See scripts/_diag_15patches.py's run (not kept in
// the repo) for the full per-patch comparison table. Used both as
// ANALYZE's default scene (App.tsx) and the landing page's hero visual
// (pages/Landing.tsx) -- the same real scene the product opens on.
export const PREFERRED_DEFAULT_SCENE_ID = "S2B_MSIL2A_20180511T100029_N9999_R122_T34VDM_81_44";
