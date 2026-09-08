"""Hand-built arrays where the answer is obvious, for evidence/ops.py."""

import numpy as np
import pytest

from evidence import ops

LAND = 0
WATER = 1
URBAN = 2
FOREST = 3

GSD_10M = {"gsd_metres": 10.0}
GSD_30M = {"gsd_metres": 30.0}


def _three_water_blobs() -> np.ndarray:
    """3 separate 2x2 (=4 pixel) water blobs on a 10x10 land raster, each
    pair at least 2 empty pixels apart so they can never be merged into one
    connected component."""
    mask = np.full((10, 10), LAND, dtype=int)
    mask[1:3, 1:3] = WATER  # blob 1: rows 1-2, cols 1-2
    mask[1:3, 5:7] = WATER  # blob 2: rows 1-2, cols 5-6
    mask[6:8, 1:3] = WATER  # blob 3: rows 6-7, cols 1-2
    return mask


def test_count_three_separate_water_blobs():
    mask = _three_water_blobs()
    assert ops.count(mask, WATER, min_area_m2=0, metadata=GSD_10M) == 3


def test_count_drops_components_below_min_area():
    mask = _three_water_blobs()
    # each blob is 4 px * 100 m2/px = 400 m2 -- a 500 m2 threshold drops all of them
    assert ops.count(mask, WATER, min_area_m2=500, metadata=GSD_10M) == 0
    # a 300 m2 threshold keeps all of them
    assert ops.count(mask, WATER, min_area_m2=300, metadata=GSD_10M) == 3


def test_count_no_matching_class_is_zero():
    mask = _three_water_blobs()
    assert ops.count(mask, FOREST, min_area_m2=0, metadata=GSD_10M) == 0


def test_presence_true_when_big_enough():
    mask = _three_water_blobs()
    assert ops.presence(mask, WATER, min_area_m2=400, metadata=GSD_10M) is True


def test_presence_false_when_class_absent():
    mask = _three_water_blobs()
    assert ops.presence(mask, FOREST, min_area_m2=0, metadata=GSD_10M) is False


def test_presence_false_when_below_min_area():
    mask = _three_water_blobs()
    # every blob is 400 m2; nothing clears a 1000 m2 bar
    assert ops.presence(mask, WATER, min_area_m2=1000, metadata=GSD_10M) is False


def test_size_reads_gsd_from_metadata_not_hardcoded():
    mask = np.full((4, 5), LAND, dtype=int)
    mask[0, 0:3] = URBAN  # 3 px
    mask[1, 0:3] = URBAN  # 3 px -> 6 px total

    # 6 px * (10 m)^2 = 600 m2 = 0.06 ha
    assert ops.size(mask, URBAN, GSD_10M) == pytest.approx(0.06)
    # same raster, different metadata -> different answer, proving gsd is
    # read from metadata and never hardcoded
    # 6 px * (30 m)^2 = 5400 m2 = 0.54 ha
    assert ops.size(mask, URBAN, GSD_30M) == pytest.approx(0.54)


def test_size_zero_when_class_absent():
    mask = np.full((4, 5), LAND, dtype=int)
    assert ops.size(mask, URBAN, GSD_10M) == 0.0


def _two_blocks(gap_cols: int) -> np.ndarray:
    """class_a (1) at cols 0-1, class_b (2) starting `gap_cols` empty
    columns later, both 2 cols wide, on a 2-row raster."""
    b_start = 2 + gap_cols
    mask = np.full((2, b_start + 2), LAND, dtype=int)
    mask[:, 0:2] = WATER  # class_a
    mask[:, b_start:b_start + 2] = URBAN  # class_b
    return mask


def test_adjacency_true_when_blocks_touch_at_distance_zero():
    mask = _two_blocks(gap_cols=0)  # blocks are immediately next to each other
    assert ops.adjacency(mask, WATER, URBAN, distance_m=0, metadata=GSD_10M) is True


def test_adjacency_false_when_gap_exceeds_distance():
    mask = _two_blocks(gap_cols=2)  # nearest pixels are 3 columns apart (30 m)
    # 20 m only buys 2 pixels of dilation -- not enough to bridge 3
    assert ops.adjacency(mask, WATER, URBAN, distance_m=20, metadata=GSD_10M) is False


def test_adjacency_true_when_distance_covers_gap():
    mask = _two_blocks(gap_cols=2)  # nearest pixels are 3 columns apart (30 m)
    assert ops.adjacency(mask, WATER, URBAN, distance_m=30, metadata=GSD_10M) is True


def test_adjacency_false_for_unrelated_classes():
    mask = _two_blocks(gap_cols=0)
    assert ops.adjacency(mask, WATER, FOREST, distance_m=1000, metadata=GSD_10M) is False
