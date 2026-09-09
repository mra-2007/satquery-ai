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


# --- class_id as a list -- agent.vocabulary.resolve_noun's many-to-many case -

FOREST_B = 4  # a second, distinct "forest-like" class id, alongside FOREST=3


def _two_forest_subtypes_touching() -> np.ndarray:
    """A single visually-contiguous 4x2 forest patch made of TWO different
    raw classes (FOREST and FOREST_B) side by side -- the exact shape of
    the real bug: a scene whose forest happens to be entirely one raw
    subtype must not read as "no forest" just because a different generic
    "forest" mapping picked the other subtype."""
    mask = np.full((10, 10), LAND, dtype=int)
    mask[2:6, 2:4] = FOREST     # left half of the patch
    mask[2:6, 4:6] = FOREST_B   # right half of the patch, touching the first
    return mask


def test_size_with_a_list_sums_every_class_in_it():
    mask = _two_forest_subtypes_touching()
    assert ops.size(mask, [FOREST, FOREST_B], metadata=GSD_10M) == pytest.approx(0.16)  # 16 px total


def test_size_with_a_list_is_nonzero_even_if_only_one_member_class_is_present():
    # The exact reported bug: resolve_noun("forest") used to pick ONE
    # representative class; if a scene had none of THAT class, size()
    # answered 0 even with plenty of a sibling forest class on screen.
    mask = np.full((10, 10), LAND, dtype=int)
    mask[2:6, 2:6] = FOREST  # only FOREST present, zero pixels of FOREST_B
    assert ops.size(mask, [FOREST, FOREST_B], metadata=GSD_10M) == pytest.approx(0.16)


def test_count_unions_before_labeling_not_after():
    # FOREST and FOREST_B touch, forming ONE connected patch. Counting
    # each raw class separately and summing would say "2"; unioned first,
    # it's correctly "1".
    mask = _two_forest_subtypes_touching()
    assert ops.count(mask, [FOREST, FOREST_B], min_area_m2=0, metadata=GSD_10M) == 1
    # proof the two are genuinely different raw classes, not a typo:
    assert ops.count(mask, FOREST, min_area_m2=0, metadata=GSD_10M) == 1
    assert ops.count(mask, FOREST_B, min_area_m2=0, metadata=GSD_10M) == 1


def test_presence_with_a_list():
    mask = np.full((10, 10), LAND, dtype=int)
    mask[2:4, 2:4] = FOREST_B
    assert ops.presence(mask, [FOREST, FOREST_B], min_area_m2=0, metadata=GSD_10M) is True
    assert ops.presence(mask, [FOREST], min_area_m2=1_000_000, metadata=GSD_10M) is False


def test_adjacency_with_list_class_ids_on_both_sides():
    mask = _two_blocks(gap_cols=0)  # WATER and URBAN touching directly
    assert ops.adjacency(mask, [WATER, FOREST], [URBAN], distance_m=0, metadata=GSD_10M) is True
    assert ops.adjacency(mask, [FOREST], [URBAN], distance_m=0, metadata=GSD_10M) is False


# --- buffer: the dilation primitive adjacency() itself is built on -----------


def test_buffer_at_zero_distance_still_grows_by_one_pixel_touching_semantics():
    # matches adjacency()'s own pre-existing "touching counts as distance
    # zero" behaviour (max(1, ...) iterations) -- buffer(distance_m=0) is
    # a 1-pixel dilation, not a no-op identical to class_mask.
    mask = _two_blocks(gap_cols=0)  # blocks already touch directly
    buffered = ops.buffer(mask, WATER, distance_m=0, metadata=GSD_10M)
    assert np.any(buffered & ops.class_mask(mask, URBAN))
    assert not np.array_equal(buffered, ops.class_mask(mask, WATER))  # strictly larger than the raw mask


def test_buffer_grows_by_whole_pixels_rounded_up():
    mask = _two_blocks(gap_cols=2)  # nearest pixels 3 columns (30 m) apart
    # 20 m -> ceil(20/10) = 2 px of growth -- not enough to bridge 3 columns
    assert not np.any(ops.buffer(mask, WATER, distance_m=20, metadata=GSD_10M) & ops.class_mask(mask, URBAN))
    # 30 m -> ceil(30/10) = 3 px of growth -- exactly enough
    assert np.any(ops.buffer(mask, WATER, distance_m=30, metadata=GSD_10M) & ops.class_mask(mask, URBAN))


def test_buffer_never_hardcodes_the_gsd():
    mask = _two_blocks(gap_cols=2)  # a fixed 3-pixel gap, regardless of GSD
    # 30 m at 10 m GSD -> 3 px of growth -- exactly enough to bridge the gap
    assert np.any(ops.buffer(mask, WATER, distance_m=30, metadata=GSD_10M) & ops.class_mask(mask, URBAN))
    # the SAME 30 m at 30 m GSD -> only 1 px of growth -- not enough for the
    # same 3-pixel gap. Different answers from the same distance_m, purely
    # because metadata['gsd_metres'] differs, proves the math reads it
    # rather than assuming a fixed pixel step.
    assert not np.any(ops.buffer(mask, WATER, distance_m=30, metadata=GSD_30M) & ops.class_mask(mask, URBAN))


def test_adjacency_is_expressible_purely_via_buffer():
    # documents the actual relationship: adjacency() is buffer() plus one
    # intersection test, not independently-implemented dilation logic.
    mask = _two_blocks(gap_cols=2)
    for distance_m in (0, 20, 30, 1000):
        expected = ops.adjacency(mask, WATER, URBAN, distance_m=distance_m, metadata=GSD_10M)
        actual = bool(np.any(ops.buffer(mask, WATER, distance_m, GSD_10M) & ops.class_mask(mask, URBAN)))
        assert actual == expected


# --- intersect_area: cross-raster area, e.g. "how much cropland is flooded" --

CROPLAND = FOREST  # reuse an existing id as a stand-in "cropland" class for these tests


def test_intersect_area_counts_pixels_matching_both_masks():
    before = np.full((10, 10), LAND, dtype=int)
    before[0:4, 0:4] = CROPLAND  # 16 px cropland in the BEFORE date
    after = np.full((10, 10), LAND, dtype=int)
    after[0:2, 0:2] = WATER  # only the top-left 4 px of that area is now water

    # 4 px * (10 m)^2 = 400 m2 = 0.04 ha
    assert ops.intersect_area(before, CROPLAND, after, WATER, GSD_10M) == pytest.approx(0.04)


def test_intersect_area_zero_when_regions_dont_overlap():
    before = np.full((10, 10), LAND, dtype=int)
    before[0:2, 0:2] = CROPLAND
    after = np.full((10, 10), LAND, dtype=int)
    after[8:10, 8:10] = WATER  # disjoint corner -- no pixel is both
    assert ops.intersect_area(before, CROPLAND, after, WATER, GSD_10M) == 0.0


def test_intersect_area_full_overlap_equals_the_smaller_regions_own_size():
    before = np.full((10, 10), LAND, dtype=int)
    before[0:5, 0:5] = CROPLAND  # 25 px
    after = np.full((10, 10), LAND, dtype=int)
    after[0:3, 0:3] = WATER  # 9 px, entirely inside the cropland footprint
    assert ops.intersect_area(before, CROPLAND, after, WATER, GSD_10M) == pytest.approx(
        ops.size(after, WATER, GSD_10M)
    )


def test_intersect_area_supports_list_class_ids_on_both_sides():
    before = np.full((10, 10), LAND, dtype=int)
    before[0:4, 0:4] = FOREST_B  # a different "cropland-ish" subtype
    after = np.full((10, 10), LAND, dtype=int)
    after[0:2, 0:2] = URBAN
    assert ops.intersect_area(before, [CROPLAND, FOREST_B], after, [WATER, URBAN], GSD_10M) == pytest.approx(0.04)


def test_intersect_area_reads_gsd_from_metadata_not_hardcoded():
    before = np.full((4, 5), LAND, dtype=int)
    before[0:2, 0:3] = CROPLAND  # 6 px
    after = np.full((4, 5), LAND, dtype=int)
    after[0:2, 0:3] = WATER  # same 6 px, full overlap

    assert ops.intersect_area(before, CROPLAND, after, WATER, GSD_10M) == pytest.approx(0.06)
    assert ops.intersect_area(before, CROPLAND, after, WATER, GSD_30M) == pytest.approx(0.54)


def test_intersect_area_raises_on_shape_mismatch():
    before = np.zeros((10, 10), dtype=int)
    after = np.zeros((5, 5), dtype=int)
    with pytest.raises(ValueError, match="shape mismatch"):
        ops.intersect_area(before, CROPLAND, after, WATER, GSD_10M)
