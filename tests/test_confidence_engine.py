"""Tests for confidence/engine.py: each of the five components in
isolation, the D4 rotate/flip group's own round-trip correctness, and
combine()'s weighted-average + graceful-degradation behaviour."""

from dataclasses import dataclass

import numpy as np
import pytest

from confidence import engine

RNG = np.random.default_rng(0)


def _softmax(logits: np.ndarray, axis: int = 0) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


# --- perception_margin -----------------------------------------------------


def test_perception_margin_high_when_answer_class_is_confident():
    # 4 classes, 3x3 image, every pixel overwhelmingly class 1.
    logits = np.full((4, 3, 3), -10.0)
    logits[1] = 10.0
    probs = _softmax(logits)
    margin = engine.perception_margin(probs, class_id=1, temperature=1.0)
    assert margin > 0.99


def test_perception_margin_low_when_answer_class_is_ambiguous():
    # class 0 and class 1 are nearly tied everywhere class 1 wins by a hair.
    logits = np.zeros((4, 3, 3))
    logits[1] = 0.01
    probs = _softmax(logits)
    margin = engine.perception_margin(probs, class_id=1, temperature=1.0)
    assert margin < 0.05


def test_perception_margin_defaults_to_half_when_class_absent_from_mask():
    logits = np.full((3, 2, 2), -10.0)
    logits[0] = 10.0  # every pixel is class 0
    probs = _softmax(logits)
    assert engine.perception_margin(probs, class_id=2, temperature=1.0) == 0.5


def test_perception_margin_supports_list_class_ids():
    # class 1 and class 2 pixels should both count as "the answer" -- a
    # generic noun resolved to several real classes (agent/vocabulary.py).
    logits = np.full((4, 2, 2), -10.0)
    logits[1, 0, :] = 10.0  # top row -> class 1
    logits[2, 1, :] = 10.0  # bottom row -> class 2
    probs = _softmax(logits)
    margin = engine.perception_margin(probs, class_id=[1, 2], temperature=1.0)
    assert margin > 0.99


def test_perception_margin_applies_the_given_temperature():
    logits = np.zeros((3, 2, 2))
    logits[0] = 4.0  # a confidently-predicted class
    probs = _softmax(logits)
    sharp = engine.perception_margin(probs, class_id=0, temperature=0.3)
    flat = engine.perception_margin(probs, class_id=0, temperature=3.0)
    assert sharp > flat


# --- resolution_suitability --------------------------------------------------


def test_resolution_suitability_scores_below_half_when_sub_resolution():
    # building: 10 m typical size, 10 m GSD -> min_resolvable = 25 m,
    # ratio = 10/25 = 0.4 -> score = 0.2.
    classes = {0: "building"}
    score = engine.resolution_suitability(0, classes, {"gsd_metres": 10.0})
    assert score == pytest.approx(0.2, abs=1e-9)


def test_resolution_suitability_scores_full_when_comfortably_resolvable():
    # forest_patch: 200 m typical size, 10 m GSD -> min_resolvable = 25 m,
    # ratio = 8.0 -> clipped to 1.0.
    classes = {0: "forest_patch"}
    score = engine.resolution_suitability(0, classes, {"gsd_metres": 10.0})
    assert score == pytest.approx(1.0, abs=1e-9)


def test_resolution_suitability_defaults_to_one_for_an_unregistered_class():
    classes = {0: "Arable land"}  # not in TYPICAL_OBJECT_SIZE_M
    assert engine.resolution_suitability(0, classes, {"gsd_metres": 10.0}) == 1.0


def test_resolution_suitability_defaults_to_one_for_a_list_class_id():
    classes = {0: "building", 1: "water_body"}
    assert engine.resolution_suitability([0, 1], classes, {"gsd_metres": 10.0}) == 1.0


def test_resolution_suitability_defaults_to_one_when_class_id_or_classes_is_none():
    assert engine.resolution_suitability(None, {0: "building"}, {"gsd_metres": 10.0}) == 1.0
    assert engine.resolution_suitability(0, None, {"gsd_metres": 10.0}) == 1.0


def test_resolution_suitability_degrades_as_gsd_coarsens():
    classes = {0: "water_body"}  # 30 m typical size
    fine = engine.resolution_suitability(0, classes, {"gsd_metres": 5.0})    # min_resolvable=12.5, ratio=2.4->1.0
    coarse = engine.resolution_suitability(0, classes, {"gsd_metres": 30.0})  # min_resolvable=75, ratio=0.4->0.2
    assert fine > coarse


# --- plan_validity -----------------------------------------------------------


def test_plan_validity_first_try_is_perfect():
    assert engine.plan_validity(1) == 1.0


def test_plan_validity_decays_per_retry():
    assert engine.plan_validity(2) == pytest.approx(0.8)
    assert engine.plan_validity(3) == pytest.approx(0.6)


def test_plan_validity_floors_at_the_documented_minimum():
    assert engine.plan_validity(10) == engine.PLAN_VALIDITY_FLOOR


# --- the D4 group: apply/invert must round-trip exactly -----------------------


@pytest.mark.parametrize("k,flip", engine._D4_ELEMENTS)
def test_d4_apply_then_invert_round_trips_exactly(k, flip):
    original = RNG.integers(0, 5, size=(3, 7, 11))  # deliberately non-square
    transformed = engine._apply_d4(original, k, flip)
    restored = engine._invert_d4(transformed, k, flip, spatial_axes=(1, 2))
    np.testing.assert_array_equal(restored, original)


def test_d4_has_eight_distinct_elements():
    assert len(engine._D4_ELEMENTS) == 8
    assert len(set(engine._D4_ELEMENTS)) == 8


def test_d4_transforms_are_not_all_identical_on_an_asymmetric_pattern():
    # Sanity check that _apply_d4 actually does something for at least one
    # non-identity element -- a bug that made every transform a no-op
    # would still pass the round-trip test above.
    original = np.zeros((1, 4, 4), dtype=int)
    original[0, 0, 0] = 1  # a single marked corner
    results = {(k, flip): engine._apply_d4(original, k, flip).tobytes() for k, flip in engine._D4_ELEMENTS}
    assert len(set(results.values())) > 1


# --- tta_stability -------------------------------------------------------------


@dataclass
class _FakeResult:
    mask: np.ndarray


def test_tta_stability_is_perfect_when_every_orientation_agrees(monkeypatch):
    # A fixed, count-preserving mask: 50 of 100 pixels are class 1.
    fixed_mask = np.zeros((10, 10), dtype=int)
    fixed_mask[:5, :] = 1

    def fake_segment_image(stack, sensor, *, session, config):
        return _FakeResult(mask=fixed_mask)

    monkeypatch.setattr("perception.infer.segment_image", fake_segment_image)
    stack = np.zeros((16, 10, 10), dtype=np.float32)

    score = engine.tta_stability(stack, "fused", class_id=1, metadata={}, session=None, config=None)
    assert score == pytest.approx(1.0)


def test_tta_stability_scores_zero_when_wildly_unstable(monkeypatch):
    # Alternate between "class 1 is the whole image" and "class 1 is
    # absent" across the 8 calls -- as unstable as a binary mask gets.
    calls = {"n": 0}
    full_mask = np.ones((10, 10), dtype=int)
    empty_mask = np.zeros((10, 10), dtype=int)

    def fake_segment_image(stack, sensor, *, session, config):
        calls["n"] += 1
        return _FakeResult(mask=full_mask if calls["n"] % 2 == 0 else empty_mask)

    monkeypatch.setattr("perception.infer.segment_image", fake_segment_image)
    stack = np.zeros((16, 10, 10), dtype=np.float32)

    score = engine.tta_stability(
        stack, "fused", class_id=1, metadata={}, session=None, config=None, max_relative_std=0.15,
    )
    assert score == 0.0


def test_tta_stability_is_perfect_when_the_class_is_never_detected(monkeypatch):
    empty_mask = np.zeros((10, 10), dtype=int)
    monkeypatch.setattr("perception.infer.segment_image", lambda *a, **k: _FakeResult(mask=empty_mask))
    stack = np.zeros((16, 10, 10), dtype=np.float32)

    score = engine.tta_stability(stack, "fused", class_id=1, metadata={}, session=None, config=None)
    assert score == 1.0


def test_tta_stability_calls_segment_image_exactly_eight_times(monkeypatch):
    calls = {"n": 0}
    mask = np.zeros((4, 4), dtype=int)

    def fake_segment_image(stack, sensor, *, session, config):
        calls["n"] += 1
        return _FakeResult(mask=mask)

    monkeypatch.setattr("perception.infer.segment_image", fake_segment_image)
    engine.tta_stability(
        np.zeros((16, 4, 4), dtype=np.float32), "fused", class_id=0, metadata={}, session=None, config=None,
    )
    assert calls["n"] == 8


# --- cross_source_agreement (thin wrapper over evidence.fusion) -------------


def test_cross_source_agreement_returns_fuse_evidence_agreement_fraction(monkeypatch):
    @dataclass
    class _FakeFusionResult:
        agreement_fraction: float

    captured = {}

    def fake_fuse_evidence(stack, metadata, class_id, *, session, config):
        captured.update(stack=stack, metadata=metadata, class_id=class_id, session=session, config=config)
        return _FakeFusionResult(agreement_fraction=0.75)

    monkeypatch.setattr("evidence.fusion.fuse_evidence", fake_fuse_evidence)
    stack = np.zeros((16, 4, 4), dtype=np.float32)
    metadata = {"gsd_metres": 10.0}

    result = engine.cross_source_agreement(stack, metadata, class_id=3, session="sess", config="cfg")

    assert result == 0.75
    assert captured["class_id"] == 3
    assert captured["session"] == "sess"
    assert captured["config"] == "cfg"


# --- combine() / ConfidenceResult --------------------------------------------


def test_combine_is_a_plain_weighted_average_when_everything_is_available():
    components = {
        "perception_margin": 0.9,
        "tta_stability": 0.8,
        "cross_source_agreement": 0.7,
        "resolution_suitability": 1.0,
        "plan_validity": 1.0,
    }
    result = engine.combine(components)
    expected = sum(engine.DEFAULT_WEIGHTS[k] * v for k, v in components.items())
    assert result.calibrated == pytest.approx(expected)
    assert result.weights_used == pytest.approx(engine.DEFAULT_WEIGHTS)


def test_combine_renormalizes_when_a_component_is_missing():
    # Only resolution_suitability (weight 0.15) and plan_validity (0.15)
    # available -- their weights should renormalize to 0.5/0.5.
    components = {
        "perception_margin": None,
        "tta_stability": None,
        "cross_source_agreement": None,
        "resolution_suitability": 0.8,
        "plan_validity": 0.4,
    }
    result = engine.combine(components)
    assert result.weights_used == pytest.approx({"resolution_suitability": 0.5, "plan_validity": 0.5})
    assert result.calibrated == pytest.approx(0.5 * 0.8 + 0.5 * 0.4)


def test_combine_raises_when_nothing_is_available():
    with pytest.raises(ValueError):
        engine.combine({name: None for name in engine.DEFAULT_WEIGHTS})


def test_combine_flags_refused_below_threshold():
    components = {name: 0.1 for name in engine.DEFAULT_WEIGHTS}
    result = engine.combine(components)
    assert result.calibrated < engine.REFUSAL_THRESHOLD
    assert result.refused is True


def test_combine_does_not_flag_refused_above_threshold():
    components = {name: 0.9 for name in engine.DEFAULT_WEIGHTS}
    result = engine.combine(components)
    assert result.refused is False


def test_combine_reports_the_constant_geometry_confidence():
    result = engine.combine({name: 0.9 for name in engine.DEFAULT_WEIGHTS})
    assert result.geometry_confidence == engine.DETERMINISTIC_GEOMETRY_CONFIDENCE == 1.0


def test_should_refuse_matches_the_documented_threshold():
    assert engine.should_refuse(engine.REFUSAL_THRESHOLD - 0.01) is True
    assert engine.should_refuse(engine.REFUSAL_THRESHOLD) is False
    assert engine.should_refuse(engine.REFUSAL_THRESHOLD + 0.01) is False


# --- compute_confidence() orchestration --------------------------------------


def test_compute_confidence_uses_only_available_signals():
    logits = np.full((3, 4, 4), -10.0)
    logits[1] = 10.0
    probs = _softmax(logits)

    inputs = engine.ConfidenceInputs(
        class_id=1, metadata={"gsd_metres": 10.0}, classes={1: "Arable land"}, probabilities=probs,
    )
    result = engine.compute_confidence(inputs)

    assert result.components["perception_margin"] is not None
    assert result.components["resolution_suitability"] is not None
    assert result.components["tta_stability"] is None  # no stack given
    assert result.components["cross_source_agreement"] is None  # no stack given
    assert result.components["plan_validity"] is None  # attempts not given
    assert set(result.weights_used) == {"perception_margin", "resolution_suitability"}


def test_compute_confidence_includes_plan_validity_when_attempts_given():
    inputs = engine.ConfidenceInputs(
        class_id=1, metadata={"gsd_metres": 10.0}, classes=None, plan_attempts=2,
    )
    result = engine.compute_confidence(inputs)
    assert result.components["plan_validity"] == pytest.approx(0.8)
    assert "plan_validity" in result.weights_used


def test_compute_confidence_runs_tta_and_agreement_when_stack_and_session_given(monkeypatch):
    mask = np.zeros((4, 4), dtype=int)
    monkeypatch.setattr("perception.infer.segment_image", lambda *a, **k: _FakeResult(mask=mask))

    @dataclass
    class _FakeFusionResult:
        agreement_fraction: float

    monkeypatch.setattr("evidence.fusion.fuse_evidence", lambda *a, **k: _FakeFusionResult(agreement_fraction=0.6))

    inputs = engine.ConfidenceInputs(
        class_id=0, metadata={"gsd_metres": 10.0}, classes=None,
        stack=np.zeros((16, 4, 4), dtype=np.float32), session="s", config="c",
    )
    result = engine.compute_confidence(inputs)

    assert result.components["tta_stability"] is not None
    assert result.components["cross_source_agreement"] == pytest.approx(0.6)
