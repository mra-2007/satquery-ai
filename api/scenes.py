"""Loads data/demo_patches/*.npz into queryable scenes: runs inference
once per patch (sensor_id="fused" -- these are genuine Sentinel-1+2
stacks with real SAR, per CLAUDE.md/scripts/demo_real.py), saves the
predicted mask to disk, and records scene metadata in SQLite.

Idempotent and synchronous, not a background job (no Celery, per
CLAUDE.md): ensure_scenes_loaded() is called once from api/main.py's
startup lifespan, and a restart reuses an already-loaded scene's mask
file and DB row instead of re-running inference.

Also renders the two PNGs the map needs -- a true-color preview (from the
stack's real B04/B03/B02 bands, per-channel percentile-stretched purely
for display, the same way any satellite image viewer contrast-stretches
raw reflectance) and a colorized mask -- computed on request rather than
cached, since a 120x120 PNG is cheap enough that caching would be
premature. Nothing here is fabricated: every pixel comes from the real
stack or the real predicted mask.

Also registers UPLOADED scenes (single, cross_modal pair, or change pair)
after raster_io/validate.py has passed them -- see register_single_scene/
register_cross_modal_pair/register_change_pair. An uploaded image almost
never has the model's real 16 Sentinel-1/2 channels; build_model_input_
from_bands() maps whatever bands it does have into the model's correct
optical band positions (never a naive 1:1 copy -- see scripts/demo_real.py
for why that would be wrong) and returns an explicit warning disclosing
the gap, rather than silently pretending 16 real channels were used.
"""

import io
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from agent.vocabulary import SEGMENTATION_CLASSES
from api.database import get_connection
from perception import infer
from raster_io.readers import RasterImage

ROOT_DIR = Path(__file__).resolve().parent.parent
DEMO_PATCHES_DIR = ROOT_DIR / "data" / "demo_patches"
MASKS_DIR = ROOT_DIR / "data" / "api" / "masks"
STACKS_DIR = ROOT_DIR / "data" / "api" / "stacks"

GSD_METRES = 10.0  # reBEN's own documented Sentinel-2 resolution
SENSOR = "fused"
N_CLASSES = len(SEGMENTATION_CLASSES)

# See tools/cross_modal.py: channels 0-9 are the real Sentinel-2 optical
# bands, 10-11 the two SAR (VV/VH) bands, 12-15 auxiliary.
RGB_TO_DN_SCALE = 3000.0 / 255.0  # see eval/run_rsvqa.py's docstring for this approximation's origin


def _insert_scene(
    conn,
    *,
    scene_id: str,
    filename: str,
    width: int,
    height: int,
    gsd_metres: float,
    sensor: str,
    mask_path: str,
    classes: dict[int, str],
    kind: str = "single",
    stack_path: str | None = None,
    before_mask_path: str | None = None,
    before_stack_path: str | None = None,
    warnings: list[str] | None = None,
    source: str = "demo",
    before_date: str | None = None,
    after_date: str | None = None,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO scenes "
        "(id, filename, width, height, gsd_metres, sensor, mask_path, classes_json, created_at, "
        " kind, stack_path, before_mask_path, before_stack_path, warnings_json, source, "
        " before_date, after_date) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            scene_id, filename, width, height, gsd_metres, sensor, mask_path, json.dumps(classes),
            datetime.now(timezone.utc).isoformat(), kind, stack_path, before_mask_path, before_stack_path,
            json.dumps(warnings or []), source, before_date, after_date,
        ),
    )
    conn.commit()


def new_scene_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def ensure_scenes_loaded(*, limit: int | None = None) -> None:
    """Scan DEMO_PATCHES_DIR and make sure every patch has a scenes row
    and a saved mask file. Patches already loaded (row present AND mask
    file present) are skipped -- the model is only loaded lazily (see
    perception.infer.get_shared_session()), the first time it's actually
    needed, and shared with every other demo-scene seeder in this module
    rather than each creating its own session.

    `limit`, when given, stops after registering that many NEW scenes in
    THIS call (already-registered ones are skipped for free and don't
    count against it) -- used by api/main.py's startup lifespan to seed
    just one demo scene cheaply on a memory-constrained deployment,
    without paying for every patch's inference before the app has even
    bound its port. The unlimited call happens lazily instead, the first
    time GET /scenes is requested (api/main.py's list_scenes()) -- calling
    this again there for already-registered patches costs nothing beyond
    the existence checks.
    """
    if not DEMO_PATCHES_DIR.exists():
        return
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    session = None
    config = None
    registered = 0
    conn = get_connection()
    try:
        for npz_path in sorted(DEMO_PATCHES_DIR.glob("*.npz")):
            if limit is not None and registered >= limit:
                break

            scene_id = npz_path.stem
            mask_path = MASKS_DIR / f"{scene_id}.npy"

            existing = conn.execute("SELECT id FROM scenes WHERE id = ?", (scene_id,)).fetchone()
            if existing is not None and mask_path.exists():
                continue

            if session is None:
                session, config = infer.get_shared_session()

            data = np.load(npz_path, allow_pickle=True)
            stack = data["stack"][:16].astype(np.float32)  # drop the always-zero 17th band
            result = infer.segment_image(stack, SENSOR, session=session, config=config)
            np.save(mask_path, result.mask)

            height, width = result.mask.shape
            _insert_scene(
                conn, scene_id=scene_id, filename=npz_path.name, width=width, height=height,
                gsd_metres=GSD_METRES, sensor=SENSOR, mask_path=str(mask_path),
                classes=dict(enumerate(SEGMENTATION_CLASSES)), kind="single", source="demo",
            )
            registered += 1
    finally:
        conn.close()


def ensure_demo_cross_modal_scene_loaded() -> None:
    """Seeds exactly one demo 'cross_modal'-kind scene from
    data/demo_patches/*.npz, so the CROSS-MODAL COMPARISON view (OPTICAL/
    SAR/FUSED toggles) is demonstrable without first requiring a user to
    /upload a real optical+SAR pair.

    These patches already carry genuine Sentinel-1 SAR alongside their
    Sentinel-2 optical bands in one 16-channel stack (see this module's own
    docstring and tools/cross_modal.py's) -- unlike ensure_demo_change_scene_loaded,
    which must extract a bi-temporal pair from a separate dataset, this only
    needs to register a second scene row, of kind='cross_modal' rather than
    'single', pointing at the same .npz. `mask_path` holds the fused
    segmentation (sensor_id="fused"), matching every other scene's
    AFTER/primary-mask convention; the CROSS-MODAL COMPARISON view computes
    the optical-only and SAR-only masks itself, on request, via
    GET /scenes/{id}/cross-modal-mask.

    Idempotent, called once from api/main.py's startup lifespan alongside
    ensure_scenes_loaded() and ensure_demo_change_scene_loaded(): does
    nothing once a source='demo' kind='cross_modal' scene already exists.
    Picks the first .npz in sorted order -- deterministic, same convention
    tests/test_cross_modal.py's own integration tests use (_demo_patches[0]) --
    not cherry-picked for a flattering result."""
    if not DEMO_PATCHES_DIR.exists():
        return

    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT id FROM scenes WHERE source = 'demo' AND kind = 'cross_modal' LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    if existing is not None:
        return

    patches = sorted(DEMO_PATCHES_DIR.glob("*.npz"))
    if not patches:
        return
    npz_path = patches[0]

    stack = load_scene_stack(npz_path.name)
    session, config = infer.get_shared_session()
    result = infer.segment_image(stack, SENSOR, session=session, config=config)

    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    scene_id = new_scene_id("crossmodal-demo")
    mask_path = MASKS_DIR / f"{scene_id}.npy"
    np.save(mask_path, result.mask)

    height, width = result.mask.shape
    conn = get_connection()
    try:
        _insert_scene(
            conn, scene_id=scene_id, filename=npz_path.name, width=width, height=height,
            gsd_metres=GSD_METRES, sensor=SENSOR, mask_path=str(mask_path),
            classes=dict(enumerate(SEGMENTATION_CLASSES)), kind="cross_modal", source="demo",
        )
    finally:
        conn.close()


def load_scene_mask(mask_path: str) -> np.ndarray:
    return np.load(mask_path)


def load_scene_stack(filename: str) -> np.ndarray:
    data = np.load(DEMO_PATCHES_DIR / filename, allow_pickle=True)
    return data["stack"][:16].astype(np.float32)


def load_scene_stack_for_row(row: Any) -> np.ndarray | None:
    """The 16-channel stack behind a scenes row, for tools (cross_modal)
    that need raw pixels rather than a precomputed mask. Uploaded scenes
    saved their constructed stack directly (there's no original .npz to
    re-derive it from); demo scenes reconstruct it on demand from
    data/demo_patches/*.npz, same as before. Returns None only if
    neither source is available.

    For a 'change'-kind scene this is the AFTER (current) stack -- the
    same one `mask_path` segments -- matching load_scene_mask's own
    AFTER-by-default convention; see load_scene_before_stack_for_row for
    the BEFORE date."""
    if row["stack_path"]:
        return np.load(row["stack_path"])
    if row["source"] == "demo":
        return load_scene_stack(row["filename"])
    return None


def load_scene_before_stack_for_row(row: Any) -> np.ndarray | None:
    """The BEFORE date's 16-channel stack for a 'change'-kind scene (see
    register_change_pair), for rendering its true-color preview. None for
    any scene that isn't a 'change' pair."""
    if row["before_stack_path"]:
        return np.load(row["before_stack_path"])
    return None


# A hand-picked cartographic palette, one RGB triple per SEGMENTATION_CLASSES
# index -- deep blues for water, dark greens for forest, olive/tan for
# agriculture, warm greys for built-up land, teal for wetlands. Replaces an
# earlier HSV rainbow sweep (colorsys.hsv_to_rgb(class_id / N_CLASSES, ...))
# that produced magenta/pink hues for several class ids and read as a
# generic, ungrounded AI-tool palette rather than a map.
_CLASS_COLORS: list[tuple[int, int, int]] = [
    (140, 130, 120),  # 0  Urban fabric -- warm grey
    (110, 100, 95),   # 1  Industrial or commercial units -- darker warm grey
    (196, 174, 108),  # 2  Arable land -- tan
    (176, 154, 88),   # 3  Permanent crops -- darker tan
    (150, 168, 90),   # 4  Pastures -- olive
    (168, 158, 96),   # 5  Complex cultivation patterns -- olive-tan
    (182, 172, 120),  # 6  Land principally occupied by agriculture ... -- light olive-tan
    (110, 140, 90),   # 7  Agro-forestry areas -- olive-green
    (34, 94, 42),     # 8  Broad-leaved forest -- dark green
    (21, 71, 52),     # 9  Coniferous forest -- deep blue-green
    (60, 110, 55),    # 10 Mixed forest -- medium green
    (176, 190, 120),  # 11 Natural grassland and sparsely vegetated areas -- pale olive-green
    (139, 120, 79),   # 12 Moors, heathland and sclerophyllous vegetation -- brownish olive
    (99, 129, 77),    # 13 Transitional woodland, shrub -- green-brown
    (216, 197, 150),  # 14 Beaches, dunes, sands -- tan/sand
    (72, 141, 137),   # 15 Inland wetlands -- teal
    (49, 120, 118),   # 16 Coastal wetlands -- darker teal
    (31, 89, 156),    # 17 Inland waters -- deep blue
    (16, 59, 110),    # 18 Marine waters -- darker deep blue
]
assert len(_CLASS_COLORS) == N_CLASSES


def class_color(class_id: int) -> tuple[int, int, int]:
    """A fixed, deterministic color per class_id, used everywhere this
    project renders a mask for the ANALYZE screen (render_mask_png, the
    /scenes/{id}/legend endpoint). Cartographic, not a rainbow sweep: no
    magenta, purple, or pink anywhere in _CLASS_COLORS. (scripts/demo_real.py
    is a separate offline debug script and still uses its own HSV sweep for
    quick side-by-side pred/gt PNGs -- it is never shown in the product UI.)"""
    return _CLASS_COLORS[class_id]


def render_mask_png(mask: np.ndarray) -> bytes:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id in np.unique(mask):
        rgb[mask == class_id] = class_color(int(class_id))
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


# The one accent colour reserved for "this pixel changed" -- the same
# restrained cyan the frontend's theme uses for an active data layer
# (web/src/styles/theme.css's --signal-active), never reused as a land-cover
# class colour, so a changed-pixel overlay is never confusable with a class.
_CHANGE_HIGHLIGHT_RGB = (77, 209, 224)


def render_change_mask_png(change_mask: np.ndarray) -> bytes:
    """`change_mask`: boolean raster, True where the class changed between
    dates (evidence.change.change_mask). Rendered as RGBA -- fully
    transparent where unchanged, solid highlight colour where changed --
    so the frontend can lay it directly over the AFTER image as a toggleable
    overlay, the same way the class mask already overlays via opacity."""
    rgba = np.zeros((*change_mask.shape, 4), dtype=np.uint8)
    rgba[change_mask] = (*_CHANGE_HIGHLIGHT_RGB, 235)
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


def _percentile_stretch(channel: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Per-channel contrast stretch for DISPLAY only -- never fed back
    into the model. Real reflectance values are usually too narrow a
    slice of the 0-255 range to look like anything without this, exactly
    like every other satellite image viewer's "stretch" setting."""
    lo, hi = np.percentile(channel, [low, high])
    if hi <= lo:
        return np.zeros_like(channel, dtype=np.uint8)
    stretched = np.clip((channel - lo) / (hi - lo), 0.0, 1.0)
    return (stretched * 255).astype(np.uint8)


def render_preview_png(stack: np.ndarray) -> bytes:
    """True-color preview from the stack's real B04/B03/B02 bands (see
    scripts/demo_real.py's docstring for the channel-order verification:
    channel 0 = B02 Blue, 1 = B03 Green, 2 = B04 Red)."""
    red = _percentile_stretch(stack[2])
    green = _percentile_stretch(stack[1])
    blue = _percentile_stretch(stack[0])
    rgb = np.stack([red, green, blue], axis=-1)
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


# --- uploaded scenes: honest channel mapping, then registration ------------


def build_model_input_from_bands(data: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """`data`: (band_count, H, W) raw pixel array, band-first (rasterio's
    own convention). Returns a (16, H, W) float32 model input plus any
    disclosed accuracy warnings.

    An uploaded image essentially never has the model's real 16
    Sentinel-1/2 channels. A 3-band upload is treated as true-color RGB
    and mapped into the model's actual B04/B03/B02 positions -- NOT a
    naive channels[:3] = data copy, which would silently swap Red and
    Blue (see scripts/demo_real.py's docstring for the channel-order
    verification this mirrors). Any other band count is placed in order
    starting at channel 0. Either way the gap is disclosed, never hidden.
    """
    warnings: list[str] = []
    band_count, height, width = data.shape
    channels = np.zeros((16, height, width), dtype=np.float32)

    if band_count == 3:
        reflectance = data.astype(np.float32) * RGB_TO_DN_SCALE
        channels[2] = reflectance[0]  # B04 Red   <- band 0 (R)
        channels[1] = reflectance[1]  # B03 Green <- band 1 (G)
        channels[0] = reflectance[2]  # B02 Blue  <- band 2 (B)
        warnings.append(
            "Only 3 of the model's 16 expected channels are real (RGB, mapped to its "
            "B04/B03/B02 positions); the remaining 13 are zero-filled. Accuracy is "
            "reduced -- treat results as approximate."
        )
    elif band_count >= 16:
        channels[:] = data[:16].astype(np.float32)
        if band_count > 16:
            warnings.append(f"Uploaded image has {band_count} channels; only the first 16 were used.")
    else:
        channels[:band_count] = data.astype(np.float32)
        warnings.append(
            f"Only {band_count} of the model's 16 expected channels are real (placed in "
            f"order starting at channel 0); the remaining {16 - band_count} are "
            "zero-filled. Accuracy is reduced -- treat results as approximate."
        )
    return channels, warnings


def place_sar_bands(channels: np.ndarray, sar_data: np.ndarray) -> list[str]:
    """Places `sar_data`'s bands into the model's two real SAR positions
    (channel 10 = VV, 11 = VH), modifying `channels` in place. Unlike the
    optical RGB case, there is no principled scale to convert arbitrary
    uploaded pixel intensities into calibrated dB backscatter, so none is
    attempted -- raw values are used as-is, and that gap is disclosed
    rather than silently assumed correct."""
    warnings: list[str] = []
    band_count = sar_data.shape[0]
    if band_count >= 2:
        channels[10] = sar_data[0].astype(np.float32)
        channels[11] = sar_data[1].astype(np.float32)
        if band_count > 2:
            warnings.append(f"SAR image has {band_count} channels; only the first 2 (used as VV, VH) were used.")
    elif band_count == 1:
        channels[10] = sar_data[0].astype(np.float32)
        warnings.append("SAR image has only 1 channel (used as VV); VH is zero-filled.")
    warnings.append(
        "SAR channel values are raw uploaded pixel intensities, not calibrated dB "
        "backscatter -- there is no principled way to recover that from an arbitrary "
        "upload. SAR- and fused-sensor results for this scene are approximate."
    )
    return warnings


def register_single_scene(image: RasterImage, session: Any, config: Any) -> tuple[str, list[str]]:
    """Build a model input from whatever bands `image` has, segment it
    (sensor_id="optical" -- an uploaded single image never has real SAR),
    and register a new 'single'-kind scene. Returns (scene_id, warnings)."""
    channels, warnings = build_model_input_from_bands(image.data)
    result = infer.segment_image(channels, "optical", session=session, config=config)

    scene_id = new_scene_id("upload")
    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    STACKS_DIR.mkdir(parents=True, exist_ok=True)
    mask_path = MASKS_DIR / f"{scene_id}.npy"
    stack_path = STACKS_DIR / f"{scene_id}.npy"
    np.save(mask_path, result.mask)
    np.save(stack_path, channels)

    height, width = result.mask.shape
    conn = get_connection()
    try:
        _insert_scene(
            conn, scene_id=scene_id, filename=image.path.name, width=width, height=height,
            gsd_metres=image.gsd_metres, sensor="optical", mask_path=str(mask_path),
            classes=dict(enumerate(SEGMENTATION_CLASSES)), kind="single",
            stack_path=str(stack_path), warnings=warnings, source="upload",
        )
    finally:
        conn.close()
    return scene_id, warnings


def register_cross_modal_pair(
    optical_image: RasterImage, sar_image: RasterImage, session: Any, config: Any,
) -> tuple[str, list[str]]:
    """A co-registered optical+SAR pair (already phase-correlation
    checked by the caller): combine them into one real-dual-sensor stack
    and register a 'cross_modal'-kind scene, segmented with
    sensor_id="fused" so its primary mask reflects both sensors."""
    channels, warnings = build_model_input_from_bands(optical_image.data)
    warnings = warnings + place_sar_bands(channels, sar_image.data)

    result = infer.segment_image(channels, "fused", session=session, config=config)

    scene_id = new_scene_id("crossmodal")
    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    STACKS_DIR.mkdir(parents=True, exist_ok=True)
    mask_path = MASKS_DIR / f"{scene_id}.npy"
    stack_path = STACKS_DIR / f"{scene_id}.npy"
    np.save(mask_path, result.mask)
    np.save(stack_path, channels)

    height, width = result.mask.shape
    conn = get_connection()
    try:
        _insert_scene(
            conn, scene_id=scene_id, filename=optical_image.path.name, width=width, height=height,
            gsd_metres=optical_image.gsd_metres, sensor="fused", mask_path=str(mask_path),
            classes=dict(enumerate(SEGMENTATION_CLASSES)), kind="cross_modal",
            stack_path=str(stack_path), warnings=warnings, source="upload",
        )
    finally:
        conn.close()
    return scene_id, warnings


def register_change_pair(
    before_image: RasterImage, after_image: RasterImage, session: Any, config: Any,
    *, source: str = "upload", before_date: str | None = None, after_date: str | None = None,
) -> tuple[str, list[str]]:
    """A co-registered bi-temporal pair: segment each date independently
    (sensor_id="optical") and register a 'change'-kind scene. `mask_path`
    holds the AFTER (current) state -- so ordinary count/size/presence/
    adjacency/caption/ground questions answer about "now" by default --
    and `before_mask_path` holds the earlier date, for the 'change' tool.
    Both dates' 16-channel model inputs are saved (`stack_path` = after,
    `before_stack_path` = before) so the CHANGE COMPARISON view can render
    a true-color preview of either date. `source` is "upload" for a real
    bi-temporal /upload and "demo" for api.scenes.ensure_demo_change_scene_loaded's
    seeded OSCD pair -- both go through this exact same function.
    `before_date`/`after_date` are free-text, caller-supplied acquisition
    dates (never invented here) -- None whenever the caller doesn't know
    one, which the CHANGE COMPARISON view shows as a plain "BEFORE"/"AFTER"
    label rather than a fabricated date."""
    before_channels, before_warnings = build_model_input_from_bands(before_image.data)
    after_channels, after_warnings = build_model_input_from_bands(after_image.data)
    warnings = before_warnings + after_warnings

    before_result = infer.segment_image(before_channels, "optical", session=session, config=config)
    after_result = infer.segment_image(after_channels, "optical", session=session, config=config)

    scene_id = new_scene_id("change")
    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    STACKS_DIR.mkdir(parents=True, exist_ok=True)
    before_mask_path = MASKS_DIR / f"{scene_id}-before.npy"
    mask_path = MASKS_DIR / f"{scene_id}-after.npy"
    before_stack_path = STACKS_DIR / f"{scene_id}-before.npy"
    stack_path = STACKS_DIR / f"{scene_id}-after.npy"
    np.save(before_mask_path, before_result.mask)
    np.save(mask_path, after_result.mask)
    np.save(before_stack_path, before_channels)
    np.save(stack_path, after_channels)

    height, width = after_result.mask.shape
    conn = get_connection()
    try:
        _insert_scene(
            conn, scene_id=scene_id, filename=after_image.path.name, width=width, height=height,
            gsd_metres=after_image.gsd_metres, sensor="optical", mask_path=str(mask_path),
            classes=dict(enumerate(SEGMENTATION_CLASSES)), kind="change",
            stack_path=str(stack_path), before_mask_path=str(before_mask_path),
            before_stack_path=str(before_stack_path), warnings=warnings, source=source,
            before_date=before_date, after_date=after_date,
        )
    finally:
        conn.close()
    return scene_id, warnings


# --- Demo change-pair scene, seeded from data/oscd/ -------------------------

OSCD_DIR = ROOT_DIR / "data" / "oscd"
OSCD_EXTRACTED_DIR = OSCD_DIR / "_extracted"
# data/oscd/test.parquet row 7: the most visually demonstrable of the 24
# pairs across both its splits (~10% of pixels flagged changed by OSCD's
# own ground-truth mask -- the largest fraction of any row inspected),
# picked once by inspecting every row's changed-pixel fraction, not chosen
# to flatter this pipeline's own output.
OSCD_DEMO_SPLIT = "test"
OSCD_DEMO_ROW_INDEX = 7


def ensure_demo_change_scene_loaded() -> None:
    """Seeds exactly one demo 'change'-kind scene from data/oscd/, real
    bi-temporal Sentinel-2 optical imagery (the Onera Satellite Change
    Detection dataset), so the CHANGE COMPARISON view is demonstrable
    without first requiring a user to /upload a bi-temporal pair.

    Idempotent, called once from api/main.py's startup lifespan alongside
    ensure_scenes_loaded(): does nothing once a source='demo' kind='change'
    scene already exists. Goes through the exact same
    raster_io.validate.validate_images -> register_change_pair path a real
    bi-temporal /upload does -- OSCD's own PNGs are extracted to disk once
    and read back with read_image() like any other upload, not special-cased.
    OSCD's own per-pixel change-ground-truth mask is NOT used (it has no
    class labels -- a binary "changed"/"not changed" raster, not this
    model's 19-class vocabulary); this scene answers questions from its own
    real predicted before/after masks like any other scene."""
    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT id FROM scenes WHERE source = 'demo' AND kind = 'change' LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    if existing is not None:
        return

    parquet_path = OSCD_DIR / f"{OSCD_DEMO_SPLIT}.parquet"
    if not parquet_path.exists():
        # Same defensive guard as ensure_scenes_loaded()'s own "if not
        # DEMO_PATCHES_DIR.exists(): return" -- data/oscd/ is the large
        # (39 MB) OSCD source archive, deliberately excluded from a lean
        # deployment image (Dockerfile only copies data/demo_patches/), so
        # this one demo 'change' scene simply isn't seeded there rather
        # than crashing FastAPI's whole startup lifespan over one missing
        # optional demo.
        return

    import pandas as pd  # local import: only this one-time seeding path needs parquet support

    from raster_io.readers import read_image
    from raster_io.validate import ImageInput, validate_images

    df = pd.read_parquet(parquet_path)
    row = df.iloc[OSCD_DEMO_ROW_INDEX]

    OSCD_EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    before_path = OSCD_EXTRACTED_DIR / "demo_change_before.png"
    after_path = OSCD_EXTRACTED_DIR / "demo_change_after.png"
    before_path.write_bytes(row["image1"]["bytes"])
    after_path.write_bytes(row["image2"]["bytes"])

    # OSCD's PNGs carry no georeferencing (like any benchmark crop -- see
    # raster_io/readers.py); OSCD is itself built on Sentinel-2, so 10 m is
    # this dataset's own documented resolution, not an assumed default.
    before_image = read_image(before_path, gsd_metres_override=GSD_METRES)
    after_image = read_image(after_path, gsd_metres_override=GSD_METRES)

    result = validate_images(
        [ImageInput(before_image, "optical"), ImageInput(after_image, "optical")],
        expected_count=2, require_matching_crs=False,
    )
    if not result.ok or result.fallback_single_modality:
        return  # OSCD ships pre-registered pairs -- should never actually happen

    session, config = infer.get_shared_session()
    register_change_pair(before_image, after_image, session, config, source="demo")
