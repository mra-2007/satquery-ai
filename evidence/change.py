"""Bi-temporal change detection over two class-ID rasters of the same area.

The LLM never computes these numbers -- see CLAUDE.md: "No pixel, no claim."
Area is always derived from metadata['gsd_metres'], never hardcoded.
"""

from dataclasses import dataclass

import numpy as np


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


def detect_change(
    before: np.ndarray,
    after: np.ndarray,
    metadata: dict,
    class_names: dict[int, str] | None = None,
) -> ChangeReport:
    """Full bi-temporal change report: change mask, per-class area deltas, and
    a plain-English summary."""
    return ChangeReport(
        mask=change_mask(before, after),
        area_changes=area_changes(before, after, metadata),
        summary=summarize(before, after, metadata, class_names),
    )
