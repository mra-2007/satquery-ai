"""Deterministic geometry over class-ID rasters.

The LLM never computes these numbers -- see CLAUDE.md: "No pixel, no claim."
Every function takes the raster's metadata dict and reads gsd_metres from
it; GSD is never hardcoded to 10 m or any other value.

Every `class_id` (and `class_a`/`class_b`) parameter accepts either a
single int or a list of ints. This matters for real questions:
agent/vocabulary.py's resolve_noun() maps a generic noun like "forest" or
"water" to ALL of its real matching classes (Broad-leaved/Coniferous/Mixed
forest; Inland/Marine waters) rather than picking one arbitrarily -- a
scene that's 97% Broad-leaved forest and 0% Mixed forest used to answer
"0 hectares of forest" if the noun happened to resolve to just "Mixed
forest". A list of ids is unioned into one boolean mask (np.isin) BEFORE
any measurement, so count() also correctly merges adjacent pixels of
different classes in the list into one connected component -- "how many
forest patches" doesn't fragment one real patch at every raw-class
boundary inside it.
"""

import math

import numpy as np
from scipy import ndimage

ClassId = int | list[int]


def _pixel_area_m2(metadata: dict) -> float:
    gsd = metadata["gsd_metres"]
    return gsd * gsd


def class_mask(mask: np.ndarray, class_id: ClassId) -> np.ndarray:
    """Boolean raster, True where `mask` matches `class_id` -- a single
    class, or the union of every class in a list."""
    if isinstance(class_id, (list, tuple, set)):
        return np.isin(mask, list(class_id))
    return mask == class_id


def _component_areas_m2(mask: np.ndarray, class_id: ClassId, metadata: dict) -> np.ndarray:
    binary = class_mask(mask, class_id)
    labeled, n_components = ndimage.label(binary)
    if n_components == 0:
        return np.array([], dtype=float)
    sizes = np.atleast_1d(ndimage.sum(binary, labeled, index=range(1, n_components + 1)))
    return sizes * _pixel_area_m2(metadata)


def presence(mask: np.ndarray, class_id: ClassId, min_area_m2: float, metadata: dict) -> bool:
    """True iff at least one connected component of class_id has area >= min_area_m2."""
    return count(mask, class_id, min_area_m2, metadata) > 0


def count(mask: np.ndarray, class_id: ClassId, min_area_m2: float, metadata: dict) -> int:
    """Number of connected components of class_id with area >= min_area_m2."""
    areas = _component_areas_m2(mask, class_id, metadata)
    return int(np.sum(areas >= min_area_m2))


def size(mask: np.ndarray, class_id: ClassId, metadata: dict) -> float:
    """Total area covered by class_id, in hectares."""
    pixel_count = int(np.sum(class_mask(mask, class_id)))
    return pixel_count * _pixel_area_m2(metadata) / 10_000.0


def adjacency(mask: np.ndarray, class_a: ClassId, class_b: ClassId, distance_m: float, metadata: dict) -> bool:
    """True iff any pixel of class_a lies within distance_m of a pixel of class_b."""
    gsd = metadata["gsd_metres"]
    iterations = max(1, math.ceil(distance_m / gsd))
    dilated_a = ndimage.binary_dilation(class_mask(mask, class_a), iterations=iterations)
    return bool(np.any(dilated_a & class_mask(mask, class_b)))
