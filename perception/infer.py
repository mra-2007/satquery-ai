"""Runs models/satquery_model.onnx on CPU via onnxruntime.

The model is fixed-size: it only accepts exactly 120x120 spatial tiles
(image: (1, 16, 120, 120) float32, sensor_id: (1,) int64 where 0=optical,
1=sar, 2=fused), returning segmentation logits (1, 19, 120, 120) and
classification logits (1, 19). Any larger image is tiled into overlapping
120x120 windows, run tile by tile, and stitched back with feathered
blending -- so an object straddling a tile boundary gets one smooth,
continuous probability field (and therefore one connected mask region,
not two separately-counted fragments) rather than a hard seam.

Per CLAUDE.md: CPU inference only (CPUExecutionProvider is forced, never
left to onnxruntime's default provider selection). Per "No pixel, no
claim": this module returns both the argmax mask AND the full softmax
probabilities, so perception/confidence.py can compute a real margin
(top1 - top2 probability) rather than inventing a confidence number.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

PERCEPTION_DIR = Path(__file__).resolve().parent
ROOT_DIR = PERCEPTION_DIR.parent
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "satquery_model.onnx"
DEFAULT_CONFIG_PATH = ROOT_DIR / "models" / "class_config.json"

TILE_SIZE = 120
OVERLAP_FRACTION = 0.25

SENSOR_IDS = {"optical": 0, "sar": 1, "fused": 2}


class InferenceError(ValueError):
    """Raised for a malformed input (wrong channel count, unknown sensor)."""


@dataclass
class ClassConfig:
    """models/class_config.json, loaded once and reused across calls."""

    classmap: dict[int, int]
    nclasses: int
    labels: list[str]
    use_ch: list[int]
    mean: np.ndarray  # (len(use_ch),)
    std: np.ndarray   # (len(use_ch),)


@dataclass
class InferenceResult:
    mask: np.ndarray           # (H, W) int -- argmax class id per pixel
    probabilities: np.ndarray  # (nclasses, H, W) float32 -- softmax, sums to 1 over axis 0
    labels: list[str]
    # (nclasses,) float32 -- the model's separate scene-level classification
    # head, sigmoid-activated (confirmed empirically against a real demo
    # patch: sigmoid probabilities for the classes actually present in the
    # segmentation mask were all >0.85, softmax crushed everything but the
    # single largest class -- correct for BigEarthNet's multi-label scene
    # tagging, where more than one class is routinely true at once).
    # Indexed by `labels` (this ClassConfig's own alphabetical order) --
    # NOT by SEGMENTATION_CLASSES/`mask`'s order. See
    # agent/vocabulary.py's module docstring and evidence/fusion.py for why
    # conflating the two is a real, previously-made mistake in this project.
    classification: np.ndarray
    transform: Any = None      # passed through unchanged, for georeferencing


def load_class_config(path: Path = DEFAULT_CONFIG_PATH) -> ClassConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ClassConfig(
        classmap={int(k): v for k, v in data["classmap"].items()},
        nclasses=data["nclasses"],
        labels=data["labels"],
        use_ch=data["use_ch"],
        mean=np.asarray(data["mean"], dtype=np.float64),
        std=np.asarray(data["std"], dtype=np.float64),
    )


def load_model(path: Path = DEFAULT_MODEL_PATH) -> ort.InferenceSession:
    """CPU only, per CLAUDE.md -- never leave provider selection to
    onnxruntime's default (which would pick a GPU provider if present)."""
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _softmax(logits: np.ndarray, axis: int) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def _normalize_tile(tile: np.ndarray, config: ClassConfig) -> np.ndarray:
    """Select the model's `use_ch` channels (in order) and apply the
    matching per-channel (x - mean) / std."""
    if tile.shape[0] <= max(config.use_ch):
        raise InferenceError(
            f"image has {tile.shape[0]} channel(s); use_ch requires index {max(config.use_ch)}"
        )
    selected = tile[config.use_ch, :, :].astype(np.float64)
    mean = config.mean[:, None, None]
    std = config.std[:, None, None]
    return (selected - mean) / std


def _infer_tile(
    session: ort.InferenceSession, tile: np.ndarray, sensor_id: int, config: ClassConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Run one exactly-(C, 120, 120) tile through the model. Returns
    (segmentation softmax probabilities, shape (nclasses, 120, 120);
    classification sigmoid probabilities, shape (nclasses,))."""
    normalized = _normalize_tile(tile, config).astype(np.float32)
    batch = normalized[None, :, :, :]
    sensor_arr = np.array([sensor_id], dtype=np.int64)

    outputs = session.run(
        ["segmentation", "classification"], {"image": batch, "sensor_id": sensor_arr}
    )
    seg_logits = outputs[0][0]   # (nclasses, 120, 120)
    cls_logits = outputs[1][0]   # (nclasses,)
    return _softmax(seg_logits, axis=0), _sigmoid(cls_logits)


def _tile_starts(size: int, tile_size: int, stride: int) -> list[int]:
    """Start offsets covering [0, size) with `tile_size`-wide windows,
    `stride` apart, always including a final window flush with the far
    edge (so the whole dimension is covered, even if size doesn't divide
    evenly by stride)."""
    if size <= tile_size:
        return [0]
    starts = list(range(0, size - tile_size + 1, stride))
    if starts[-1] != size - tile_size:
        starts.append(size - tile_size)
    return starts


def _feather_weight(tile_size: int, overlap_px: int) -> np.ndarray:
    """A (tile_size, tile_size) blend weight: 1.0 in the core, ramping
    linearly to a small positive value over the `overlap_px`-wide band at
    each edge. Combined with dividing the stitched mosaic by its summed
    weight, this is what makes overlapping tiles blend smoothly instead of
    producing a seam (or duplicating an object that straddles a
    boundary)."""
    ramp = np.ones(tile_size, dtype=np.float64)
    if overlap_px > 0:
        up = (np.arange(overlap_px) + 1) / overlap_px
        ramp[:overlap_px] = up
        ramp[tile_size - overlap_px:] = up[::-1]
    return np.outer(ramp, ramp)


def _extract_tile(image: np.ndarray, row0: int, col0: int, tile_size: int) -> tuple[np.ndarray, int, int]:
    """The (channels, tile_size, tile_size) window at (row0, col0),
    zero-padded if it runs past the image's edge. Returns (tile, valid_h,
    valid_w) -- the padded region is never counted when stitching."""
    channels, height, width = image.shape
    valid_h = min(tile_size, height - row0)
    valid_w = min(tile_size, width - col0)
    tile = np.zeros((channels, tile_size, tile_size), dtype=image.dtype)
    tile[:, :valid_h, :valid_w] = image[:, row0:row0 + valid_h, col0:col0 + valid_w]
    return tile, valid_h, valid_w


def segment_image(
    image: np.ndarray,
    sensor: str,
    *,
    session: ort.InferenceSession,
    config: ClassConfig,
    transform: Any = None,
) -> InferenceResult:
    """Segment `image` (channels, H, W), tiling with 25% overlap and
    feathered blending whenever it's larger than one 120x120 tile.
    `transform` is passed through unchanged -- the output has the same
    (H, W) shape and pixel grid as the input, so the same georeferencing
    still applies without modification.
    """
    if sensor not in SENSOR_IDS:
        raise InferenceError(f"unknown sensor {sensor!r}; expected one of {sorted(SENSOR_IDS)}")
    sensor_id = SENSOR_IDS[sensor]

    _channels, height, width = image.shape
    overlap_px = int(TILE_SIZE * OVERLAP_FRACTION)
    stride = TILE_SIZE - overlap_px

    row_starts = _tile_starts(height, TILE_SIZE, stride)
    col_starts = _tile_starts(width, TILE_SIZE, stride)
    weight_2d = _feather_weight(TILE_SIZE, overlap_px)

    prob_accum = np.zeros((config.nclasses, height, width), dtype=np.float64)
    weight_accum = np.zeros((height, width), dtype=np.float64)
    # Classification is a single scene-level (not per-pixel) vector, so
    # tiles are combined with one scalar weight each (its valid pixel
    # area) rather than the segmentation head's per-pixel feather field --
    # a tile that's mostly real image (not padding) says more about the
    # whole scene than a tile that's mostly zero-padded edge.
    classification_accum = np.zeros(config.nclasses, dtype=np.float64)
    classification_weight = 0.0

    for row0 in row_starts:
        for col0 in col_starts:
            tile, valid_h, valid_w = _extract_tile(image, row0, col0, TILE_SIZE)
            tile_seg_probs, tile_cls_probs = _infer_tile(session, tile, sensor_id, config)

            prob_accum[:, row0:row0 + valid_h, col0:col0 + valid_w] += (
                tile_seg_probs[:, :valid_h, :valid_w] * weight_2d[:valid_h, :valid_w]
            )
            weight_accum[row0:row0 + valid_h, col0:col0 + valid_w] += weight_2d[:valid_h, :valid_w]

            tile_weight = float(valid_h * valid_w)
            classification_accum += tile_cls_probs * tile_weight
            classification_weight += tile_weight

    probabilities = (prob_accum / weight_accum[None, :, :]).astype(np.float32)
    mask = np.argmax(probabilities, axis=0).astype(np.int64)
    classification = (classification_accum / classification_weight).astype(np.float32)

    return InferenceResult(
        mask=mask, probabilities=probabilities, labels=config.labels,
        classification=classification, transform=transform,
    )


def run_inference(
    image: np.ndarray,
    sensor: str,
    *,
    transform: Any = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
    session: ort.InferenceSession | None = None,
    config: ClassConfig | None = None,
) -> InferenceResult:
    """Convenience entry point: loads the model/config (or reuses the ones
    passed in -- pass both when calling this repeatedly, loading an ONNX
    session is not cheap) and segments `image`."""
    session = session or load_model(model_path)
    config = config or load_class_config(config_path)
    return segment_image(image, sensor, session=session, config=config, transform=transform)
