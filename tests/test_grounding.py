"""Tests for tools/grounding.py: resolving a referring expression to one
connected region of a class-ID mask via agent/vocabulary.py, and
returning its polygon/bbox in pixel and (when available) geographic
coordinates.

No LLM is involved anywhere in tools/grounding.py -- unlike
tools/caption.py, there is no Gemini call to force offline in these
tests. This file is still safe to run with no network access at all."""

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

from agent.vocabulary import SEGMENTATION_CLASSES
from evidence.schema import Polygon
from tools.grounding import GroundingError, GroundingResult, ground

GSD_10M = {"gsd_metres": 10.0}

INLAND_WATERS = SEGMENTATION_CLASSES.index("Inland waters")
MARINE_WATERS = SEGMENTATION_CLASSES.index("Marine waters")
ARABLE_LAND = SEGMENTATION_CLASSES.index("Arable land")
URBAN_FABRIC = SEGMENTATION_CLASSES.index("Urban fabric")


def _two_water_bodies_mask() -> np.ndarray:
    """20x20, mostly arable land, a small (2x2=4px) water blob in the
    north-west corner and a larger (4x4=16px) one in the south-east --
    distinct enough in both position and size to tell direction-based and
    superlative-based selection apart."""
    mask = np.full((20, 20), ARABLE_LAND, dtype=int)
    mask[1:3, 1:3] = INLAND_WATERS  # north-west, small
    mask[15:19, 15:19] = INLAND_WATERS  # south-east, large
    return mask


# --- noun resolution ----------------------------------------------------------


def test_ground_resolves_noun_via_vocabulary():
    # "water body" is a generic noun that resolves to BOTH real water
    # classes (agent/vocabulary.py) -- class_id is the full list this
    # scene's mask only has one of, class_name joins both real names.
    result = ground(_two_water_bodies_mask(), GSD_10M, "the water body in the north-west")
    assert result.class_name == "Inland waters / Marine waters"
    assert result.class_id == [INLAND_WATERS, MARINE_WATERS]


def test_ground_generic_noun_still_finds_a_region_present_under_only_one_of_its_classes():
    # Regression test: a generic noun resolving to several classes must
    # still find a region when the mask only has ONE of them -- the bug
    # this whole change fixes (a picked "representative" class absent
    # from a given scene used to make count()/size() answer 0/False even
    # though the concept was clearly present under a sibling class).
    mask = np.full((10, 10), ARABLE_LAND, dtype=int)
    mask[2:4, 2:4] = MARINE_WATERS  # only Marine waters present, no Inland waters at all
    result = ground(mask, GSD_10M, "the water body")
    assert result.class_name == "Inland waters / Marine waters"
    assert result.area_ha == pytest.approx(0.04)


def test_ground_unresolvable_noun_raises():
    with pytest.raises(GroundingError, match="could not resolve"):
        ground(_two_water_bodies_mask(), GSD_10M, "the spaceship in the north")


def test_ground_resolved_class_absent_from_mask_raises():
    mask = np.full((10, 10), ARABLE_LAND, dtype=int)
    with pytest.raises(GroundingError, match="Urban fabric"):
        ground(mask, GSD_10M, "the building in the north")


# --- direction qualifiers -----------------------------------------------------


def test_north_west_selects_the_north_west_region():
    result = ground(_two_water_bodies_mask(), GSD_10M, "the water body in the north-west")
    assert result.area_ha == pytest.approx(0.04)  # the small 4 px blob
    assert result.candidate_count == 2


def test_south_east_selects_the_south_east_region():
    result = ground(_two_water_bodies_mask(), GSD_10M, "the water body in the south-east")
    assert result.area_ha == pytest.approx(0.16)  # the large 16 px blob


@pytest.mark.parametrize("phrase", ["north west", "northwest", "north-west"])
def test_north_west_phrase_variants_are_all_understood(phrase):
    result = ground(_two_water_bodies_mask(), GSD_10M, f"the water body in the {phrase}")
    assert result.qualifier["direction"] == "north-west"


def test_central_selects_the_region_closest_to_the_centre():
    mask = np.full((30, 30), ARABLE_LAND, dtype=int)
    mask[1:3, 1:3] = INLAND_WATERS  # corner
    mask[14:17, 14:17] = INLAND_WATERS  # near centre
    result = ground(mask, GSD_10M, "the central water body")
    assert result.centroid_pixel[0] == pytest.approx(15.0, abs=1.0)
    assert result.centroid_pixel[1] == pytest.approx(15.0, abs=1.0)


# --- superlative qualifiers ----------------------------------------------------


def test_largest_selects_the_biggest_region_regardless_of_position():
    result = ground(_two_water_bodies_mask(), GSD_10M, "the largest water body")
    assert result.area_ha == pytest.approx(0.16)
    assert result.qualifier["superlative"] == "largest"


def test_smallest_selects_the_smallest_region_regardless_of_position():
    result = ground(_two_water_bodies_mask(), GSD_10M, "the smallest water body")
    assert result.area_ha == pytest.approx(0.04)
    assert result.qualifier["superlative"] == "smallest"


def test_superlative_wins_over_a_direction_word_in_the_same_phrase():
    # Both qualifiers are parsed and reported (qualifier reflects what was
    # found in the text, for audit), but SELECTION prioritizes the
    # superlative -- proven here with a mask where the two rules would
    # disagree if direction were used instead: south-east is the LARGE
    # blob, yet "smallest" must still win.
    mask = _two_water_bodies_mask()
    result = ground(mask, GSD_10M, "the smallest water body in the south-east")
    assert result.qualifier == {"direction": "south-east", "superlative": "smallest"}
    assert result.area_ha == pytest.approx(0.04)  # the small NW blob, not the large SE one


def test_no_qualifier_defaults_to_the_largest_region():
    result = ground(_two_water_bodies_mask(), GSD_10M, "the water body")
    assert result.area_ha == pytest.approx(0.16)
    assert result.qualifier == {"direction": None, "superlative": None}


# --- geometry: pixel coordinates, always present -----------------------------


def test_pixel_bbox_matches_the_known_blob_extent():
    result = ground(_two_water_bodies_mask(), GSD_10M, "the water body in the north-west")
    assert result.pixel_bbox == (1, 1, 2, 2)  # (row_min, col_min, row_max, col_max)


def test_pixel_polygon_is_a_valid_polygon_covering_the_blob():
    result = ground(_two_water_bodies_mask(), GSD_10M, "the water body in the north-west")
    assert isinstance(result.pixel_polygon, Polygon)
    xs = [pt[0] for pt in result.pixel_polygon.coordinates[0]]
    ys = [pt[1] for pt in result.pixel_polygon.coordinates[0]]
    assert min(xs) == pytest.approx(1.0) and max(xs) == pytest.approx(3.0)
    assert min(ys) == pytest.approx(1.0) and max(ys) == pytest.approx(3.0)


def test_centroid_pixel_is_inside_the_selected_region():
    result = ground(_two_water_bodies_mask(), GSD_10M, "the water body in the north-west")
    row, col = result.centroid_pixel
    assert 1 <= row <= 2 and 1 <= col <= 2


# --- geometry: geographic coordinates, only when transform+crs given -------


def test_geographic_fields_are_none_without_transform_or_crs():
    result = ground(_two_water_bodies_mask(), GSD_10M, "the water body in the north-west")
    assert result.geo_bbox is None
    assert result.geo_polygon is None
    assert result.centroid_lonlat is None


def _georeferenced_metadata() -> dict:
    transform = from_origin(500000, 4649000, 10, 10)
    return {"gsd_metres": 10.0, "transform": transform, "crs": CRS.from_epsg(32643).to_string()}


def test_geographic_fields_are_populated_when_transform_and_crs_given():
    result = ground(_two_water_bodies_mask(), _georeferenced_metadata(), "the water body in the north-west")

    assert result.geo_bbox is not None
    min_lon, min_lat, max_lon, max_lat = result.geo_bbox
    assert -180 <= min_lon <= max_lon <= 180
    assert -90 <= min_lat <= max_lat <= 90

    assert result.geo_polygon is not None
    assert isinstance(result.geo_polygon, Polygon)

    assert result.centroid_lonlat is not None
    lon, lat = result.centroid_lonlat
    assert -180 <= lon <= 180
    assert -90 <= lat <= 90
    assert min_lon <= lon <= max_lon
    assert min_lat <= lat <= max_lat


def test_geographic_fields_still_none_with_only_transform_and_no_crs():
    transform = from_origin(500000, 4649000, 10, 10)
    metadata = {"gsd_metres": 10.0, "transform": transform}
    result = ground(_two_water_bodies_mask(), metadata, "the water body in the north-west")
    assert result.geo_bbox is None
    assert result.geo_polygon is None


# --- result shape ---------------------------------------------------------------


def test_ground_returns_a_grounding_result_with_qualifier_metadata():
    result = ground(_two_water_bodies_mask(), GSD_10M, "the smallest water body")
    assert isinstance(result, GroundingResult)
    assert result.candidate_count == 2
    assert result.qualifier["superlative"] == "smallest"
