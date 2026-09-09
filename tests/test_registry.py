"""Tests for agent/registry.py: tool registration, model/tool selection by
GSD+modality, parameter-schema validation, and the GSD-derived capability
table."""

import pytest

from agent.registry import (
    REGISTRY,
    CapabilityTable,
    ParameterValidationError,
    SelectionResult,
    Tool,
    capability_table,
    select_model,
    validate_parameters,
)

EXPECTED_TOOL_NAMES = {
    "segment", "count", "size", "presence", "adjacency", "change", "caption", "ground", "cross_modal",
    "verify",
}

# 16 bands: enough for every registered tool, including cross_modal (which
# needs the model's full multi-channel input, not just >=3 human-viewable
# bands like caption/ground).
SENTINEL2_OPTICAL = {"gsd_metres": 10.0, "modality": "optical", "band_count": 16}
SENTINEL1_SAR = {"gsd_metres": 10.0, "modality": "sar", "band_count": 2}


# --- Tool dataclass and registration -----------------------------------------


def test_registry_has_exactly_the_ten_required_tools():
    assert set(REGISTRY) == EXPECTED_TOOL_NAMES


def test_tool_fields_are_populated():
    tool = REGISTRY["count"]
    assert isinstance(tool, Tool)
    assert tool.name == "count"
    assert tool.description
    assert tool.accepted_modalities in ("optical", "sar", "both")
    assert len(tool.gsd_range) == 2
    assert tool.required_band_count >= 1
    assert isinstance(tool.permitted_parameters, dict)


def test_tool_rejects_invalid_modality():
    with pytest.raises(ValueError):
        Tool("x", "d", "hyperspectral", (1.0, 10.0), 1, {})


def test_tool_rejects_inverted_gsd_range():
    with pytest.raises(ValueError):
        Tool("x", "d", "optical", (10.0, 1.0), 1, {})


# --- select_model: routes by GSD and modality --------------------------------


def test_select_model_sentinel2_optical_gets_all_tools():
    result = select_model(SENTINEL2_OPTICAL)
    assert isinstance(result, SelectionResult)
    assert {t.name for t in result.tools} == EXPECTED_TOOL_NAMES


def test_select_model_sar_excludes_optical_only_tools():
    result = select_model(SENTINEL1_SAR)
    selected_names = {t.name for t in result.tools}
    assert "caption" not in selected_names
    assert "ground" not in selected_names
    # cross_modal is also excluded here, but for a different reason: a
    # real Sentinel-1-only scene has just 2 (VV/VH) bands, short of the
    # 16 cross_modal's own segmentation runs require -- not a modality
    # rejection like caption/ground's.
    assert "cross_modal" not in selected_names
    assert selected_names == EXPECTED_TOOL_NAMES - {"caption", "ground", "cross_modal"}


def test_select_model_sar_rejection_reason_mentions_modality():
    result = select_model(SENTINEL1_SAR)
    caption_entries = [e for e in result.trace if e.parameters["candidate"] == "caption"]
    assert len(caption_entries) == 1
    entry = caption_entries[0]
    assert entry.output["accepted"] is False
    assert any("modality" in reason for reason in entry.output["reasons"])


def test_select_model_coarse_gsd_excludes_only_gsd_limited_tools():
    # band_count=16 so only the GSD cap (not cross_modal's band requirement)
    # is what's under test here.
    coarse_optical = {"gsd_metres": 45.0, "modality": "optical", "band_count": 16}
    result = select_model(coarse_optical)
    selected_names = {t.name for t in result.tools}
    # caption/ground cap out at 30 m; everything else tolerates up to 60 m
    assert selected_names == EXPECTED_TOOL_NAMES - {"caption", "ground"}


def test_select_model_insufficient_bands_excludes_only_band_limited_tools():
    grayscale = {"gsd_metres": 10.0, "modality": "optical", "band_count": 1}
    result = select_model(grayscale)
    selected_names = {t.name for t in result.tools}
    # caption/ground need >=3 bands; cross_modal needs the full 16-channel
    # input; the class-raster tools need only 1
    assert selected_names == EXPECTED_TOOL_NAMES - {"caption", "ground", "cross_modal"}


def test_select_model_logs_a_decision_for_every_registered_tool():
    result = select_model(SENTINEL2_OPTICAL)
    assert len(result.trace) == len(REGISTRY)
    assert {e.parameters["candidate"] for e in result.trace} == EXPECTED_TOOL_NAMES
    for entry in result.trace:
        assert entry.task.startswith("select_model:")
        assert entry.tool == "agent.registry.select_model"
        assert entry.confidence == 1.0


# --- validate_parameters: rejects anything outside the tool's schema --------


def test_validate_parameters_accepts_a_valid_call():
    validate_parameters(REGISTRY["count"], {"class_id": 1, "min_area_m2": 500.0})  # no raise


def test_validate_parameters_rejects_unknown_parameter():
    with pytest.raises(ParameterValidationError):
        validate_parameters(REGISTRY["count"], {"class_id": 1, "bogus": True})


def test_validate_parameters_rejects_wrong_type():
    with pytest.raises(ParameterValidationError):
        validate_parameters(REGISTRY["count"], {"class_id": "one", "min_area_m2": 500.0})


def test_validate_parameters_accepts_plain_int_for_float_parameter():
    # 500 (not 500.0) is exactly what hand-written or LLM-emitted JSON for a
    # whole-number float parameter looks like -- must not be rejected.
    validate_parameters(REGISTRY["count"], {"class_id": 1, "min_area_m2": 500})  # no raise


def test_validate_parameters_rejects_bool_for_float_parameter():
    with pytest.raises(ParameterValidationError):
        validate_parameters(REGISTRY["count"], {"class_id": 1, "min_area_m2": True})


def test_validate_parameters_rejects_any_parameter_on_empty_schema_tool():
    assert REGISTRY["segment"].permitted_parameters == {}
    with pytest.raises(ParameterValidationError):
        validate_parameters(REGISTRY["segment"], {"unexpected": 1})


def test_validate_parameters_accepts_empty_call_on_empty_schema_tool():
    validate_parameters(REGISTRY["segment"], {})  # no raise


def test_validate_parameters_ground_requires_query_string():
    validate_parameters(REGISTRY["ground"], {"query": "red rooftop"})  # no raise
    with pytest.raises(ParameterValidationError):
        validate_parameters(REGISTRY["ground"], {"query": 123})


def test_validate_parameters_cross_modal_accepts_cloud_simulation_bool():
    validate_parameters(REGISTRY["cross_modal"], {})  # no raise -- cloud_simulation is optional
    validate_parameters(REGISTRY["cross_modal"], {"cloud_simulation": True})  # no raise
    with pytest.raises(ParameterValidationError):
        validate_parameters(REGISTRY["cross_modal"], {"cloud_simulation": "yes"})


def test_cross_modal_requires_the_models_full_16_channel_input():
    assert REGISTRY["cross_modal"].required_band_count == 16


# --- capability_table: computed from GSD, not hardcoded ----------------------


def test_capability_table_at_10m_buildings_not_countable_water_bodies_are():
    table = capability_table({"gsd_metres": 10.0})
    assert isinstance(table, CapabilityTable)
    assert table.classes["building"].countable is False
    assert table.classes["water_body"].countable is True


def test_capability_table_min_resolvable_is_2_5x_gsd():
    table = capability_table({"gsd_metres": 10.0})
    for capability in table.classes.values():
        assert capability.min_resolvable_m == pytest.approx(25.0)


def test_capability_table_large_classes_countable_even_when_small_ones_are_not():
    table = capability_table({"gsd_metres": 10.0})
    assert table.classes["vehicle"].countable is False
    assert table.classes["agricultural_field"].countable is True
    assert table.classes["forest_patch"].countable is True


def test_capability_table_is_recomputed_not_hardcoded_across_gsd():
    coarse = capability_table({"gsd_metres": 10.0})
    fine = capability_table({"gsd_metres": 2.0})

    # building is not countable at 10 m (10 <= 25) but is at 2 m (10 > 5)
    assert coarse.classes["building"].countable is False
    assert fine.classes["building"].countable is True

    # min_resolvable_m itself must differ -- proof it's derived, not fixed
    assert coarse.classes["building"].min_resolvable_m != fine.classes["building"].min_resolvable_m


def test_capability_table_boundary_is_strictly_greater_than():
    # vehicle (5 m) exactly equals min_resolvable_m at gsd=2 (2.5*2=5) -- not countable
    table = capability_table({"gsd_metres": 2.0})
    assert table.classes["vehicle"].typical_object_size_m == 5.0
    assert table.classes["vehicle"].min_resolvable_m == 5.0
    assert table.classes["vehicle"].countable is False


def test_capability_table_logs_a_decision_per_class():
    table = capability_table({"gsd_metres": 10.0})
    assert len(table.trace) == len(table.classes)
    for entry in table.trace:
        assert entry.task.startswith("capability_table:")
        assert entry.tool == "agent.registry.capability_table"
        assert "countable" in entry.output
        assert entry.confidence == 1.0
