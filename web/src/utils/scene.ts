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
