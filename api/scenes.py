"""Loads data/demo_patches/*.npz into queryable scenes: runs inference
once per patch (sensor_id="fused" -- these are genuine Sentinel-1+2
stacks with real SAR, per CLAUDE.md/scripts/demo_real.py), saves the
predicted mask to disk, and records scene metadata in SQLite.

Idempotent and synchronous, not a background job (no Celery, per
CLAUDE.md): ensure_scenes_loaded() is called once from api/main.py's
startup lifespan, and a restart reuses an already-loaded scene's mask
file and DB row instead of re-running inference.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from agent.vocabulary import SEGMENTATION_CLASSES
from api.database import get_connection
from perception import infer

ROOT_DIR = Path(__file__).resolve().parent.parent
DEMO_PATCHES_DIR = ROOT_DIR / "data" / "demo_patches"
MASKS_DIR = ROOT_DIR / "data" / "api" / "masks"

GSD_METRES = 10.0  # reBEN's own documented Sentinel-2 resolution
SENSOR = "fused"


def ensure_scenes_loaded() -> None:
    """Scan DEMO_PATCHES_DIR and make sure every patch has a scenes row
    and a saved mask file. Patches already loaded (row present AND mask
    file present) are skipped -- the model is only loaded lazily, the
    first time it's actually needed."""
    if not DEMO_PATCHES_DIR.exists():
        return
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    session = None
    config = None
    conn = get_connection()
    try:
        for npz_path in sorted(DEMO_PATCHES_DIR.glob("*.npz")):
            scene_id = npz_path.stem
            mask_path = MASKS_DIR / f"{scene_id}.npy"

            existing = conn.execute("SELECT id FROM scenes WHERE id = ?", (scene_id,)).fetchone()
            if existing is not None and mask_path.exists():
                continue

            if session is None:
                session = infer.load_model()
                config = infer.load_class_config()

            data = np.load(npz_path, allow_pickle=True)
            stack = data["stack"][:16].astype(np.float32)  # drop the always-zero 17th band
            result = infer.segment_image(stack, SENSOR, session=session, config=config)
            np.save(mask_path, result.mask)

            height, width = result.mask.shape
            classes = dict(enumerate(SEGMENTATION_CLASSES))
            conn.execute(
                "INSERT OR REPLACE INTO scenes "
                "(id, filename, width, height, gsd_metres, sensor, mask_path, classes_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    scene_id, npz_path.name, width, height, GSD_METRES, SENSOR,
                    str(mask_path), json.dumps(classes), datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
    finally:
        conn.close()


def load_scene_mask(mask_path: str) -> np.ndarray:
    return np.load(mask_path)
