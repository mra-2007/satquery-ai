"""Tests for api/rendering.py's build_answer/build_geometry: specifically
the "fusion" tool branch -- without it, a plan whose final step is fusion
falls through to the generic str(output) fallback, leaking a raw Python
dict repr into the UI as the answer text (a real bug this project hit and
fixed while wiring evidence/fusion.py's frontend rendering)."""

import numpy as np
import pytest

from agent.tasks import ClassificationResult
from api.rendering import build_answer, build_confidence, build_geometry
from confidence.engine import DETERMINISTIC_GEOMETRY_SOURCE
from evidence.schema import TraceStep

CLASSES = {0: "Urban fabric", 1: "Inland waters"}
GSD_10M = {"gsd_metres": 10.0}


def _water_mask() -> np.ndarray:
    mask = np.zeros((10, 10), dtype=np.int64)
    mask[2:5, 2:5] = 1  # Inland waters
    return mask


def _fusion_output(fused_present: bool) -> dict:
    return {
        "class_id": 1,
        "class_name": "Inland waters",
        "fused_present": fused_present,
        "fused_confidence": 0.76,
        "agreement": {"segmentation": True, "classification": False, "spectral_index": True, "sar": True},
        "agreement_fraction": 0.75,
        "correlated_sources": ["segmentation", "classification"],
        "weighting_note": "The segmentation and classification heads share one encoder ...",
        "summary": "Inland waters: fused verdict is PRESENT (4 of 4 sources available, 75% agree).",
        "sources": [],
    }


def _fusion_trace_step(fused_present: bool = True) -> TraceStep:
    return TraceStep(
        task="execute:f1", tool="fusion", parameters={"class_id": 1},
        output=_fusion_output(fused_present), confidence=1.0,
    )


def test_build_answer_fusion_value_is_the_fused_present_boolean():
    value, units, answer_text, limitation = build_answer([_fusion_trace_step(True)], CLASSES)
    assert value is True
    assert units is None
    assert limitation is None


def test_build_answer_fusion_text_is_the_real_summary_not_a_dict_repr():
    _value, _units, answer_text, _limitation = build_answer([_fusion_trace_step(True)], CLASSES)
    assert answer_text == "Inland waters: fused verdict is PRESENT (4 of 4 sources available, 75% agree)."
    # the bug this test guards against: a raw Python dict repr leaking through
    assert not answer_text.startswith("{")


def test_build_answer_fusion_absent_verdict_is_false():
    value, _units, _text, _limitation = build_answer([_fusion_trace_step(False)], CLASSES)
    assert value is False


def test_build_geometry_fusion_vectorizes_the_checked_class():
    mask = _water_mask()
    step = _fusion_trace_step(True)
    geometry = build_geometry(mask, CLASSES, step)

    assert geometry.type == "FeatureCollection"
    assert len(geometry.features) == 1
    assert geometry.features[0].properties == {"class": "Inland waters"}


# --- metadata / conversational: same class of bug (a raw dict repr leaking
# through as the answer text) as fusion above, guarded the same way -------


def _metadata_trace_step(aspect: str = "resolution") -> TraceStep:
    return TraceStep(
        task="execute:m1", tool="metadata", parameters={"aspect": aspect},
        output={"aspect": aspect, "answer_text": "10 m ground sample distance ...", "detail": {}},
        confidence=1.0,
    )


def test_build_answer_metadata_value_and_text_are_the_real_answer_text():
    value, units, answer_text, limitation = build_answer([_metadata_trace_step()], CLASSES)
    assert value == "10 m ground sample distance ..."
    assert answer_text == value
    assert units is None
    assert limitation is None
    assert not value.startswith("{")


def test_build_geometry_metadata_falls_back_to_the_whole_scene():
    mask = _water_mask()
    geometry = build_geometry(mask, CLASSES, _metadata_trace_step())
    assert geometry.type == "FeatureCollection"
    assert geometry.features[0].properties == {"note": "whole scene extent"}


# --- intersect: "how much cropland is flooded?" -----------------------------


def _intersect_trace_step(value: float = 0.03) -> TraceStep:
    return TraceStep(
        task="execute:i1", tool="intersect", parameters={"class_a": 0, "class_b": 1},
        output=value, confidence=1.0,
    )


def test_build_answer_intersect_value_is_the_hectare_float():
    value, units, answer_text, limitation = build_answer([_intersect_trace_step(0.03)], CLASSES)
    assert value == 0.03
    assert units == "hectares"
    assert limitation is None
    assert not str(answer_text).startswith("{")


def test_build_answer_intersect_text_names_both_classes():
    _value, _units, answer_text, _limitation = build_answer([_intersect_trace_step(0.03)], CLASSES)
    assert answer_text == "0.03 ha of urban fabric is now inland waters"


def test_build_geometry_intersect_falls_back_to_the_whole_scene():
    # No before raster reaches this function -- only the whole-scene extent
    # is available, same as 'change'.
    mask = _water_mask()
    geometry = build_geometry(mask, CLASSES, _intersect_trace_step())
    assert geometry.type == "FeatureCollection"
    assert geometry.features[0].properties == {"note": "whole scene extent"}


# --- change with a class_id focus: "how much land changed to water?" -------


def _focused_change_trace_step(focus_gained_ha: float = 0.05) -> TraceStep:
    return TraceStep(
        task="execute:c1", tool="change", parameters={"class_id": 1},
        output={
            "changed_pixels": 5,
            "area_changes": {0: {"gained_ha": 0.0, "lost_ha": 0.05}, 1: {"gained_ha": 0.05, "lost_ha": 0.0}},
            "summary": "Inland waters gained 0.05 ha and lost 0.00 ha (net +0.05 ha).",
            "focus_class_id": 1,
            "focus_gained_ha": focus_gained_ha,
        },
        confidence=1.0,
    )


def _unfocused_change_trace_step() -> TraceStep:
    return TraceStep(
        task="execute:c1", tool="change", parameters={},
        output={
            "changed_pixels": 5,
            "area_changes": {0: {"gained_ha": 0.0, "lost_ha": 0.05}, 1: {"gained_ha": 0.05, "lost_ha": 0.0}},
            "summary": "Inland waters gained 0.05 ha and lost 0.00 ha (net +0.05 ha).",
            "focus_class_id": None,
            "focus_gained_ha": None,
        },
        confidence=1.0,
    )


def test_build_answer_focused_change_value_is_the_focus_gained_hectares():
    value, units, answer_text, limitation = build_answer([_focused_change_trace_step(0.05)], CLASSES)
    assert value == 0.05
    assert units == "hectares"
    assert limitation is None
    assert "0.05 ha changed to inland waters" in answer_text


def test_build_answer_unfocused_change_still_returns_the_plain_summary():
    value, units, answer_text, limitation = build_answer([_unfocused_change_trace_step()], CLASSES)
    assert value == "Inland waters gained 0.05 ha and lost 0.00 ha (net +0.05 ha)."
    assert answer_text == value
    assert units is None
    assert limitation is None


def _conversational_trace_step(kind: str = "greeting") -> TraceStep:
    return TraceStep(
        task="execute:c1", tool="conversational", parameters={"kind": kind},
        output={"kind": kind, "reply": "Hi! I'm SatQuery AI ..."},
        confidence=1.0,
    )


def test_build_answer_conversational_value_and_text_are_the_real_reply():
    value, units, answer_text, limitation = build_answer([_conversational_trace_step()], CLASSES)
    assert value == "Hi! I'm SatQuery AI ..."
    assert answer_text == value
    assert units is None
    assert limitation is None
    assert not value.startswith("{")


# --- build_confidence: classify_task + final tool, plus confidence/engine.py's
# calibrated composite and its always-1.0 deterministic-geometry entry when
# agent/executor.py recorded one -------------------------------------------


def _classification(confidence: float = 0.9) -> ClassificationResult:
    return ClassificationResult(label="vqa", confidence=confidence, trace=[])


def _presence_step(confidence: float = 1.0) -> TraceStep:
    return TraceStep(
        task="execute:s1", tool="presence", parameters={"class_id": 1, "min_area_m2": 0},
        output=True, confidence=confidence,
    )


def _confidence_step(calibrated: float = 0.61, geometry_confidence: float = 1.0) -> TraceStep:
    return TraceStep(
        task="confidence:s1", tool="confidence.engine.compute_confidence", parameters={"class_id": 1},
        output={
            "calibrated": calibrated, "components": {}, "weights_used": {},
            "geometry_confidence": geometry_confidence, "refused": calibrated < 0.55,
        },
        confidence=calibrated,
    )


def test_build_confidence_without_a_confidence_step_matches_the_old_two_entries():
    confidences = build_confidence(_classification(0.9), [_presence_step(1.0)])
    sources = [c.source for c in confidences]
    assert sources == ["agent.tasks.classify_task", "tool:presence"]
    assert confidences[0].confidence == pytest.approx(0.9)
    assert confidences[1].confidence == pytest.approx(1.0)


def test_build_confidence_with_a_confidence_step_adds_calibrated_and_geometry_entries():
    trace = [_presence_step(0.61), _confidence_step(calibrated=0.61)]
    confidences = build_confidence(_classification(0.9), trace)
    by_source = {c.source: c.confidence for c in confidences}

    assert by_source["agent.tasks.classify_task"] == pytest.approx(0.9)
    assert by_source["tool:presence"] == pytest.approx(0.61)  # the executor already overwrote this
    assert by_source["confidence_engine.calibrated"] == pytest.approx(0.61)
    assert by_source[DETERMINISTIC_GEOMETRY_SOURCE] == pytest.approx(1.0)


def test_build_confidence_geometry_entry_is_always_one_even_when_calibrated_is_low():
    trace = [_presence_step(0.2), _confidence_step(calibrated=0.2)]
    confidences = build_confidence(_classification(0.9), trace)
    by_source = {c.source: c.confidence for c in confidences}
    assert by_source["confidence_engine.calibrated"] == pytest.approx(0.2)
    assert by_source[DETERMINISTIC_GEOMETRY_SOURCE] == 1.0


def test_build_confidence_with_no_execute_step_at_all():
    confidences = build_confidence(_classification(0.7), [])
    assert [c.source for c in confidences] == ["agent.tasks.classify_task"]
