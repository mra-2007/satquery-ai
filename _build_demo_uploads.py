"""One-off script: exports three real /upload test fixtures from a real
demo_patches/*.npz stack, into demo_uploads/. Not part of the app --
run once, inspect/verify the outputs, then this script itself can be
deleted (kept out of the repo, matching CLAUDE.md's "Minimal
dependencies" -- these are fixtures, not application code).

All three items now share ONE source patch (T29SNB_24_78) -- the earlier
revision of this script had to split item 3 onto a different patch
because raster_io/validate.py's cross-modal co-registration check used
phase correlation, which is meaningless between optical and SAR (see
that module's own docstring) and happened to misfire on this patch's
own noise. Now that the check verifies dimensions/CRS instead of phase
correlation for a declared cross-sensor pair, that workaround is gone --
a pixel-perfect optical+SAR pair from the SAME patch always registers.
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from api.scenes import _percentile_stretch

GSD_METRES = 10.0
OUT_DIR = Path("demo_uploads")
OUT_DIR.mkdir(exist_ok=True)

PATCH_NAME = "S2B_MSIL2A_20180326T112109_N9999_R037_T29SNB_24_78"
# T29SNB's own MGRS tile -> UTM zone 29, latitude band S (32N-40N) ->
# EPSG:32629 (WGS84 / UTM zone 29N) -- the real zone this patch's real
# tile actually falls in (Portuguese coast), not an arbitrary placeholder.
# The origin itself is a plausible, valid point within that zone/hemisphere
# (not surveyed to this exact patch's true corner -- that would need a
# precise MGRS->UTM corner lookup this script doesn't have -- so it's
# disclosed here as "plausible", not claimed as survey-exact).
CRS_EPSG = "EPSG:32629"
ORIGIN_EASTING, ORIGIN_NORTHING = 600_000.0, 4_400_000.0

data = np.load(Path("data/demo_patches") / f"{PATCH_NAME}.npz", allow_pickle=True)
stack = data["stack"][:16].astype(np.float32)  # the model's real 16-channel input, unmodified
height, width = stack.shape[1], stack.shape[2]
transform = from_origin(ORIGIN_EASTING, ORIGIN_NORTHING, GSD_METRES, GSD_METRES)
crs = CRS.from_string(CRS_EPSG)


def _write_geotiff(path: Path, bands: np.ndarray, dtype: str) -> None:
    count = bands.shape[0]
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=count,
        dtype=dtype, crs=crs, transform=transform,
    ) as dst:
        dst.write(bands.astype(dtype))
    print(f"  wrote {path} -- {count} band(s), {dtype}, {path.stat().st_size / 1024:.0f} KiB")


# --- 1. Full 16-channel GeoTIFF -- every channel real, correct GSD/CRS -----
print("1. full 16-channel upload (expect: no warning banner)")
_write_geotiff(OUT_DIR / "01_full_16channel.tif", stack, "float32")

# --- 2. 3-band true-color RGB GeoTIFF -- triggers the "3 of 16" warning ---
# Percentile-stretched to uint8, exactly the way api/scenes.py's own
# render_preview_png() renders a true-color preview -- a REALISTIC
# "someone uploaded an RGB screenshot" upload, not a raw reflectance crop.
print("2. 3-band RGB upload (expect: '3 of 16 channels' warning)")
red = _percentile_stretch(stack[2])    # B04 Red
green = _percentile_stretch(stack[1])  # B03 Green
blue = _percentile_stretch(stack[0])   # B02 Blue
rgb = np.stack([red, green, blue], axis=0)  # (3, H, W), band-first
_write_geotiff(OUT_DIR / "02_rgb_3band.tif", rgb, "uint8")

# --- 3. Co-registered optical + SAR pair for cross-modal upload -----------
# optical file: all 16 positions, SAR positions (10=VV, 11=VH) zeroed --
# band_count==16 so build_model_input_from_bands() adds NO warning on the
# optical side; the sar file supplies the two real SAR channels, so the
# reconstructed 16-channel input ends up fully real on every channel.
# Co-registration is now verified by matching CRS/dimensions (both files
# share this script's own transform/crs and the same real pixel grid),
# not phase correlation -- see raster_io/validate.py's own docstring.
print("3. cross-modal optical+SAR pair (expect: only the standard SAR disclosure warning)")
optical_16 = stack.copy()
optical_16[10] = 0.0
optical_16[11] = 0.0
_write_geotiff(OUT_DIR / "03a_crossmodal_optical.tif", optical_16, "float32")

sar_2band = stack[10:12]  # VV, VH -- real Sentinel-1 backscatter from this same patch
_write_geotiff(OUT_DIR / "03b_crossmodal_sar.tif", sar_2band, "float32")

print("\ndone.")
