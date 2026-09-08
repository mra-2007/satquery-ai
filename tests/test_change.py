"""Hand-built before/after arrays where the answer is obvious, for
evidence/change.py."""

import numpy as np
import pytest

from evidence import change

LAND = 0
WATER = 1
URBAN = 2

GSD_10M = {"gsd_metres": 10.0}  # 100 m2/px = 0.01 ha/px
CLASS_NAMES = {LAND: "Land", WATER: "Water", URBAN: "Urban"}


def _before() -> np.ndarray:
    return np.array(
        [
            [LAND, LAND, LAND, LAND],
            [LAND, WATER, WATER, LAND],
            [LAND, WATER, WATER, LAND],
            [LAND, LAND, LAND, LAND],
        ]
    )


def _after_urban_reclaims_part_of_water() -> np.ndarray:
    # 3 of the 4 water pixels become urban; 1 stays water; land is untouched
    return np.array(
        [
            [LAND, LAND, LAND, LAND],
            [LAND, URBAN, URBAN, LAND],
            [LAND, URBAN, WATER, LAND],
            [LAND, LAND, LAND, LAND],
        ]
    )


def test_change_mask_marks_only_changed_pixels():
    mask = change.change_mask(_before(), _after_urban_reclaims_part_of_water())
    expected = np.array(
        [
            [False, False, False, False],
            [False, True, True, False],
            [False, True, False, False],
            [False, False, False, False],
        ]
    )
    np.testing.assert_array_equal(mask, expected)
    assert mask.sum() == 3


def test_change_mask_all_false_when_identical():
    before = _before()
    mask = change.change_mask(before, before.copy())
    assert not mask.any()


def test_change_mask_raises_on_shape_mismatch():
    with pytest.raises(ValueError):
        change.change_mask(_before(), np.zeros((3, 3), dtype=int))


def test_area_changes_gained_and_lost_hectares():
    changes = change.area_changes(_before(), _after_urban_reclaims_part_of_water(), GSD_10M)

    # 3 px * 0.01 ha/px = 0.03 ha
    assert changes[URBAN]["gained_ha"] == pytest.approx(0.03)
    assert changes[URBAN]["lost_ha"] == pytest.approx(0.0)

    assert changes[WATER]["lost_ha"] == pytest.approx(0.03)
    assert changes[WATER]["gained_ha"] == pytest.approx(0.0)

    # land never changed
    assert changes[LAND]["gained_ha"] == pytest.approx(0.0)
    assert changes[LAND]["lost_ha"] == pytest.approx(0.0)

    # closed system: whatever area was gained must equal what was lost elsewhere
    total_gained = sum(d["gained_ha"] for d in changes.values())
    total_lost = sum(d["lost_ha"] for d in changes.values())
    assert total_gained == pytest.approx(total_lost)


def test_area_changes_zero_when_identical():
    before = _before()
    changes = change.area_changes(before, before.copy(), GSD_10M)
    assert all(d["gained_ha"] == 0.0 and d["lost_ha"] == 0.0 for d in changes.values())


def test_area_changes_scales_with_gsd_not_hardcoded():
    changes_10m = change.area_changes(_before(), _after_urban_reclaims_part_of_water(), GSD_10M)
    changes_30m = change.area_changes(
        _before(), _after_urban_reclaims_part_of_water(), {"gsd_metres": 30.0}
    )
    # 3 px * (30 m)^2 / 10_000 = 0.27 ha, not 0.03 ha
    assert changes_30m[URBAN]["gained_ha"] == pytest.approx(0.27)
    assert changes_10m[URBAN]["gained_ha"] != changes_30m[URBAN]["gained_ha"]


def test_summarize_mentions_only_changed_classes():
    summary = change.summarize(
        _before(), _after_urban_reclaims_part_of_water(), GSD_10M, CLASS_NAMES
    )
    assert "Urban gained 0.03 ha and lost 0.00 ha (net +0.03 ha)." in summary
    assert "Water gained 0.00 ha and lost 0.03 ha (net -0.03 ha)." in summary
    assert "Land" not in summary  # land never changed, so it's omitted


def test_summarize_falls_back_to_class_id_without_names():
    summary = change.summarize(_before(), _after_urban_reclaims_part_of_water(), GSD_10M)
    assert "class 2 gained 0.03 ha" in summary


def test_summarize_no_change():
    before = _before()
    summary = change.summarize(before, before.copy(), GSD_10M, CLASS_NAMES)
    assert summary == "No change detected between the two dates."


def test_detect_change_bundles_all_three_results():
    report = change.detect_change(
        _before(), _after_urban_reclaims_part_of_water(), GSD_10M, CLASS_NAMES
    )
    assert report.mask.sum() == 3
    assert report.area_changes[URBAN]["gained_ha"] == pytest.approx(0.03)
    assert "Urban gained 0.03 ha" in report.summary
