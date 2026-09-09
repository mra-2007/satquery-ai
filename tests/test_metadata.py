"""Tests for evidence/metadata.py: filename parsing (date/platform/tile),
the real bigearthnet_txt_benchmark.csv-backed location lookup, and each of
the five metadata aspects."""

import numpy as np
import pytest

from evidence import metadata as md

REAL_FILENAME = "S2B_MSIL2A_20180511T100029_N9999_R122_T34VDM_81_44.npz"
GSD_10M = {"gsd_metres": 10.0}


def _mask() -> np.ndarray:
    m = np.zeros((10, 10), dtype=int)  # Urban fabric (class 0) everywhere by default
    m[0:5, 0:5] = 17  # Inland waters, 25 px = 2500 m2 = 0.25 ha; leaves 75 px (0.75 ha) Urban fabric
    return m


# --- filename parsing ----------------------------------------------------------


def test_parse_acquisition_date_reads_the_real_sentinel2_date_stamp():
    assert md.parse_acquisition_date(REAL_FILENAME) == "2018-05-11"


def test_parse_acquisition_date_none_for_a_filename_with_no_date_stamp():
    assert md.parse_acquisition_date("uploaded_scene.png") is None


def test_parse_platform_reads_s2a_or_s2b():
    assert md.parse_platform(REAL_FILENAME) == "S2B"
    assert md.parse_platform("S2A_MSIL2A_20170617T113321_N9999_R080_T29UPU_13_55.npz") == "S2A"


def test_parse_platform_none_for_a_non_sentinel2_filename():
    assert md.parse_platform("uploaded_scene.png") is None


def test_parse_tile_reads_the_mgrs_tile_id():
    assert md.parse_tile(REAL_FILENAME) == "34VDM"


def test_parse_tile_none_for_a_filename_with_no_tile_id():
    assert md.parse_tile("uploaded_scene.png") is None


# --- location: real bigearthnet_txt_benchmark.csv-backed lookup ----------------


@pytest.mark.skipif(not md.BENCHMARK_CSV.exists(), reason="data/benchmark/bigearthnet_txt_benchmark.csv not available")
def test_tile_location_returns_a_real_dataset_backed_location():
    location = md.tile_location(REAL_FILENAME)
    assert location is not None
    lat, lon, country = location
    # real, verified value from the benchmark CSV for tile 34VDM
    assert lat == pytest.approx(60.03, abs=0.5)
    assert lon == pytest.approx(20.14, abs=0.5)
    assert country == "Finland"


def test_tile_location_none_for_a_filename_with_no_tile():
    assert md.tile_location("uploaded_scene.png") is None


def test_tile_location_none_for_an_unrecognized_tile(monkeypatch):
    monkeypatch.setattr(md, "_tile_locations", lambda: {})
    assert md.tile_location(REAL_FILENAME) is None


# --- scene_metadata: each aspect ------------------------------------------------


def test_location_aspect_discloses_unavailable_for_a_plain_upload():
    result = md.scene_metadata(_mask(), {**GSD_10M, "filename": "uploaded_scene.png"}, "location")
    assert result.aspect == "location"
    assert result.detail["available"] is False
    assert "not available" in result.answer_text.lower() or "isn't available" in result.answer_text.lower()


@pytest.mark.skipif(not md.BENCHMARK_CSV.exists(), reason="data/benchmark/bigearthnet_txt_benchmark.csv not available")
def test_location_aspect_reports_real_country_for_a_known_tile():
    result = md.scene_metadata(_mask(), {**GSD_10M, "filename": REAL_FILENAME}, "location")
    assert result.detail["available"] is True
    assert result.detail["country"] == "Finland"
    assert "Finland" in result.answer_text


def test_date_aspect_reads_the_real_filename_date():
    result = md.scene_metadata(_mask(), {**GSD_10M, "filename": REAL_FILENAME}, "date")
    assert result.detail == {"available": True, "acquisition_date": "2018-05-11"}
    assert "2018-05-11" in result.answer_text


def test_date_aspect_unavailable_without_a_real_filename():
    result = md.scene_metadata(_mask(), {**GSD_10M, "filename": "uploaded_scene.png"}, "date")
    assert result.detail == {"available": False}


def test_sensor_aspect_reports_platform_and_sensor_mode():
    result = md.scene_metadata(_mask(), {**GSD_10M, "filename": REAL_FILENAME, "sensor": "fused"}, "sensor")
    assert result.detail == {"platform": "S2B", "sensor_mode": "fused"}
    assert "fused" in result.answer_text


def test_sensor_aspect_defaults_gracefully_without_context():
    result = md.scene_metadata(_mask(), GSD_10M, "sensor")
    assert result.detail["platform"] is None
    assert "unknown platform" in result.answer_text


def test_resolution_aspect_computes_min_resolvable_size():
    result = md.scene_metadata(_mask(), GSD_10M, "resolution")
    assert result.detail == {"gsd_metres": 10.0, "min_resolvable_m": 25.0}
    assert "25.0" in result.answer_text
    assert "10" in result.answer_text


def test_resolution_aspect_never_hardcodes_10m():
    # CLAUDE.md: "Every area calculation reads GSD from image metadata.
    # NEVER hardcode 10 m." -- a different real GSD must change the answer.
    result = md.scene_metadata(_mask(), {"gsd_metres": 20.0}, "resolution")
    assert result.detail == {"gsd_metres": 20.0, "min_resolvable_m": 50.0}


def test_classes_aspect_lists_present_classes_with_real_areas():
    result = md.scene_metadata(_mask(), GSD_10M, "classes")
    by_name = {c["class_name"]: c["area_ha"] for c in result.detail["classes"]}
    assert by_name == {"Urban fabric": pytest.approx(0.75), "Inland waters": pytest.approx(0.25)}
    assert "Urban fabric" in result.answer_text and "Inland waters" in result.answer_text


def test_classes_aspect_sorted_by_area_descending():
    mask = np.zeros((10, 10), dtype=int)
    mask[0:1, 0:9] = 17  # Inland waters, small
    mask[1:10, :] = 0    # Urban fabric, large
    result = md.scene_metadata(mask, GSD_10M, "classes")
    areas = [c["area_ha"] for c in result.detail["classes"]]
    assert areas == sorted(areas, reverse=True)


def test_unknown_aspect_raises():
    with pytest.raises(md.MetadataError, match="aspect"):
        md.scene_metadata(_mask(), GSD_10M, "weather")
