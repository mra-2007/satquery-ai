"""Bi-temporal change detection over two class-ID rasters of the same area.

The LLM never computes these numbers -- see CLAUDE.md: "No pixel, no claim."
Area is always derived from metadata['gsd_metres'], never hardcoded.
"""

from dataclasses import dataclass

import numpy as np

from evidence.ops import ClassId, class_mask


def _pixel_area_m2(metadata: dict) -> float:
    gsd = metadata["gsd_metres"]
    return gsd * gsd


def change_mask(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Boolean raster, True where the class at a pixel changed between dates."""
    if before.shape != after.shape:
        raise ValueError(f"shape mismatch: before={before.shape} after={after.shape}")
    return before != after


def area_changes(before: np.ndarray, after: np.ndarray, metadata: dict) -> dict[int, dict[str, float]]:
    """Per-class area gained and lost (hectares) between before and after."""
    pixel_area_ha = _pixel_area_m2(metadata) / 10_000.0
    class_ids = sorted(set(np.unique(before).tolist()) | set(np.unique(after).tolist()))

    result = {}
    for class_id in class_ids:
        gained_px = int(np.sum((after == class_id) & (before != class_id)))
        lost_px = int(np.sum((before == class_id) & (after != class_id)))
        result[int(class_id)] = {
            "gained_ha": gained_px * pixel_area_ha,
            "lost_ha": lost_px * pixel_area_ha,
        }
    return result


def gained_area(before: np.ndarray, after: np.ndarray, class_id: ClassId, metadata: dict) -> float:
    """Total area (hectares) that became class_id (or any class in a list
    of ids -- e.g. "water" = Inland + Marine waters) between before and
    after: pixels that were NOT any of class_id in `before` and ARE some
    member of class_id in `after`. Union-aware like every other
    evidence/ops.py tool: a pixel that changed from Inland waters to
    Marine waters was already water, so "how much land changed to water"
    must not count it -- summing each individual class's own gained_ha
    from area_changes() would (that pixel is Marine's own "gained"), which
    is why this is its own function rather than just adding up entries
    from that dict."""
    was_class = class_mask(before, class_id)
    is_class = class_mask(after, class_id)
    gained_px = int(np.sum(is_class & ~was_class))
    return gained_px * _pixel_area_m2(metadata) / 10_000.0


def summarize(
    before: np.ndarray,
    after: np.ndarray,
    metadata: dict,
    class_names: dict[int, str] | None = None,
) -> str:
    """Plain-English summary of the change between before and after."""
    class_names = class_names or {}
    changes = area_changes(before, after, metadata)

    lines = []
    for class_id, deltas in changes.items():
        gained, lost = deltas["gained_ha"], deltas["lost_ha"]
        if gained == 0 and lost == 0:
            continue
        name = class_names.get(class_id, f"class {class_id}")
        net = gained - lost
        lines.append(f"{name} gained {gained:.2f} ha and lost {lost:.2f} ha (net {net:+.2f} ha).")

    if not lines:
        return "No change detected between the two dates."
    return " ".join(lines)


@dataclass
class ChangeReport:
    mask: np.ndarray
    area_changes: dict[int, dict[str, float]]
    summary: str
    # Set only when detect_change() was called with a `class_id` focus
    # (e.g. "how much land changed to water?") -- None otherwise. Kept
    # alongside the always-computed, all-classes `area_changes` above
    # rather than replacing it, so the full per-class breakdown is never
    # lost just because one question asked about a specific class.
    focus_class_id: ClassId | None = None
    focus_gained_ha: float | None = None


def detect_change(
    before: np.ndarray,
    after: np.ndarray,
    metadata: dict,
    class_names: dict[int, str] | None = None,
    class_id: ClassId | None = None,
) -> ChangeReport:
    """Full bi-temporal change report: change mask, per-class area deltas,
    and a plain-English summary -- always computed, regardless of
    `class_id`. When `class_id` is given (e.g. "how much land changed to
    water?"), also computes that one class's own gained_area() as a
    focused scalar answer, alongside the full report."""
    return ChangeReport(
        mask=change_mask(before, after),
        area_changes=area_changes(before, after, metadata),
        summary=summarize(before, after, metadata, class_names),
        focus_class_id=class_id,
        focus_gained_ha=gained_area(before, after, class_id, metadata) if class_id is not None else None,
    )
