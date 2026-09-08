"""Deterministic geometry over class-ID rasters.

The LLM never computes these numbers -- see CLAUDE.md: "No pixel, no claim."
Every function takes the raster's metadata dict and reads gsd_metres from
it; GSD is never hardcoded to 10 m or any other value.
"""

import math

import numpy as np
from scipy import ndimage


def _pixel_area_m2(metadata: dict) -> float:
    gsd = metadata["gsd_metres"]
    return gsd * gsd


def _component_areas_m2(mask: np.ndarray, class_id: int, metadata: dict) -> np.ndarray:
    binary = mask == class_id
    labeled, n_components = ndimage.label(binary)
    if n_components == 0:
        return np.array([], dtype=float)
    sizes = np.atleast_1d(ndimage.sum(binary, labeled, index=range(1, n_components + 1)))
    return sizes * _pixel_area_m2(metadata)


def presence(mask: np.ndarray, class_id: int, min_area_m2: float, metadata: dict) -> bool:
    """True iff at least one connected component of class_id has area >= min_area_m2."""
    return count(mask, class_id, min_area_m2, metadata) > 0


def count(mask: np.ndarray, class_id: int, min_area_m2: float, metadata: dict) -> int:
    """Number of connected components of class_id with area >= min_area_m2."""
    areas = _component_areas_m2(mask, class_id, metadata)
    return int(np.sum(areas >= min_area_m2))


def size(mask: np.ndarray, class_id: int, metadata: dict) -> float:
    """Total area covered by class_id, in hectares."""
    pixel_count = int(np.sum(mask == class_id))
    return pixel_count * _pixel_area_m2(metadata) / 10_000.0


def adjacency(mask: np.ndarray, class_a: int, class_b: int, distance_m: float, metadata: dict) -> bool:
    """True iff any pixel of class_a lies within distance_m of a pixel of class_b."""
    gsd = metadata["gsd_metres"]
    iterations = max(1, math.ceil(distance_m / gsd))
    dilated_a = ndimage.binary_dilation(mask == class_a, iterations=iterations)
    return bool(np.any(dilated_a & (mask == class_b)))
