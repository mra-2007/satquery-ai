"""Tests for agent/executor.py: running a validated Plan against the real
evidence/ops.py and evidence/change.py implementations, and the resulting
ExecutionTrace's replayability."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from agent import executor as executor_module
from agent.dsl import Plan, PlanValidationError
from agent.executor import ExecutionTrace, ExecutorError, run
from agent.vocabulary import SEGMENTATION_CLASSES
from confidence.engine import ConfidenceResult, plan_validity
from evidence.change import ChangeReport
from evidence.fusion import FusionResult
from evidence.metadata import MetadataResult
from tools.caption import CaptionResult
from tools.conversational import ConversationalReply
from tools.cross_modal import CrossModalResult
from tools.grounding import GroundingResult

LAND = 0
WATER = 1
URBAN = 2
GSD_10M = {"gsd_metres": 10.0}


def _water_mask() -> np.ndarray:
    mask = np.full((10, 10), LAND, dtype=int)
    mask[1:3, 1:3] = WATER  # single 2x2 (400 m2) blob
    return mask


def _before_after() -> tuple[np.ndarray, np.ndarray]:
    before = np.array([
        [LAND, LAND, LAND, LAND],
        [LAND, WATER, WATER, LAND],
        [LAND, WATER, WATER, LAND],
        [LAND, LAND, LAND, LAND],
    ])
    after = np.array([
        [LAND, LAND, LAND, LAND],
        [LAND, URBAN, URBAN, LAND],
        [LAND, URBAN, WATER, LAND],
        [LAND, LAND, LAND, LAND],
    ])
    return before, after


def _ops_plan() -> Plan:
    return Plan.model_validate([
        {"id": "s1", "tool": "count", "parameters": {"class_id": WATER, "min_area_m2": 0}},
        {"id": "s2", "tool": "presence", "parameters": {"class_id": WATER, "min_area_m2": 1000}},
        {"id": "s3", "tool": "size", "parameters": {"class_id": WATER}},
        {"id": "s4", "tool": "adjacency", "parameters": {"class_a": WATER, "class_b": LAND, "distance_m": 0}},
    ])


# --- running evidence/ops.py tools -------------------------------------------


def test_run_executes_ops_tools_with_correct_results():
    result, trace = run(_ops_plan(), GSD_10M, mask=_water_mask())

    assert result["s1"] == 1
    assert result["s2"] is False  # 400 m2 blob, 1000 m2 threshold
    assert result["s3"] == pytest.approx(0.04)  # 4 px * 100 m2 / 10_000
    assert result["s4"] is True  # water touches land immediately

    assert isinstance(trace, ExecutionTrace)
    assert len(trace) == 4


def test_run_trace_records_task_tool_parameters_output_confidence():
    _, trace = run(_ops_plan(), GSD_10M, mask=_water_mask())
    entry = next(e for e in trace if e.tool == "count")

    assert entry.task == "execute:s1"
    assert entry.tool == "count"
    assert entry.parameters == {"class_id": WATER, "min_area_m2": 0}
    assert entry.output == 1  # scalar outputs pass through the summary unchanged
    assert entry.confidence == 1.0


def test_run_without_mask_raises_when_plan_needs_ops_tools():
    with pytest.raises(ExecutorError, match="mask"):
        run(_ops_plan(), GSD_10M)


# --- capability guardrail is applied automatically when `classes` is given --


def test_run_applies_capability_guardrail_when_classes_given():
    # s1 is 'count' on WATER; declaring it as "building" (10 m, not
    # countable at 10 m GSD) should make executor.run() rewrite it to
    # 'size' on our behalf, with no separate guardrail call required.
    result, trace = run(
        _ops_plan(), GSD_10M, mask=_water_mask(), classes={WATER: "building", LAND: "land"},
    )

    assert result["s1"] == pytest.approx(0.04)  # size in ha, not a count of 1

    guardrail_entries = [e for e in trace if e.task == "capability_guardrail"]
    assert len(guardrail_entries) == 1
    assert guardrail_entries[0].output["degraded_to"] == "size"
    assert guardrail_entries[0].parameters["class_id"] == WATER


def test_run_without_classes_skips_the_guardrail():
    result, trace = run(_ops_plan(), GSD_10M, mask=_water_mask())

    assert result["s1"] == 1  # unmodified 'count' result
    assert not any(e.task == "capability_guardrail" for e in trace)


# --- running evidence/change.py ----------------------------------------------


def test_run_executes_change_tool():
    before, after = _before_after()
    plan = Plan.model_validate([{"id": "c1", "tool": "change", "parameters": {}}])

    result, trace = run(plan, GSD_10M, before=before, after=after)

    assert isinstance(result["c1"], ChangeReport)
    assert result["c1"].mask.sum() == 3
    assert result["c1"].area_changes[URBAN]["gained_ha"] == pytest.approx(0.03)

    entry = trace.steps[0]
    assert entry.tool == "change"
    # the trace holds a SUMMARY, not the raw ChangeReport / raw mask array
    assert entry.output == {
        "changed_pixels": 3,
        "area_changes": result["c1"].area_changes,
        "summary": result["c1"].summary,
        "focus_class_id": None,
        "focus_gained_ha": None,
    }


def test_run_executes_change_tool_with_class_id_focus():
    # "how much land changed to water?" -- class_id is a plan parameter,
    # filled in via fn(**step.parameters) on top of the before/after/
    # metadata/class_names already bound by run() itself.
    before, after = _before_after()
    plan = Plan.model_validate([{"id": "c1", "tool": "change", "parameters": {"class_id": URBAN}}])

    result, trace = run(plan, GSD_10M, before=before, after=after)

    assert result["c1"].focus_class_id == URBAN
    assert result["c1"].focus_gained_ha == pytest.approx(0.03)

    entry = trace.steps[0]
    assert entry.output["focus_class_id"] == URBAN
    assert entry.output["focus_gained_ha"] == pytest.approx(0.03)


def test_run_passes_classes_through_to_change_tool_summary():
    # A real, previously-shipped bug: the 'change' tool never received the
    # scene's class_id -> name mapping, so ChangeReport.summary always read
    # "class 2 gained ..." instead of "Urban fabric gained ...", even
    # though run() already receives `classes` for the capability guardrail.
    before, after = _before_after()
    plan = Plan.model_validate([{"id": "c1", "tool": "change", "parameters": {}}])
    classes = {LAND: "Land", WATER: "Water", URBAN: "Urban fabric"}

    result, _trace = run(plan, GSD_10M, before=before, after=after, classes=classes)

    assert "Urban fabric" in result["c1"].summary
    assert "class 2" not in result["c1"].summary


def test_run_without_before_after_raises_when_plan_needs_change():
    plan = Plan.model_validate([{"id": "c1", "tool": "change", "parameters": {}}])
    with pytest.raises(ExecutorError, match="before and after"):
        run(plan, GSD_10M)


def test_run_with_only_before_raises():
    before, _after = _before_after()
    plan = Plan.model_validate([{"id": "c1", "tool": "change", "parameters": {}}])
    with pytest.raises(ExecutorError):
        run(plan, GSD_10M, before=before)


# --- running evidence/ops.py's intersect_area, over before/after ------------


def test_run_executes_intersect_tool():
    # WATER in before that is URBAN in after -- 3 px, per _before_after().
    before, after = _before_after()
    plan = Plan.model_validate([
        {"id": "i1", "tool": "intersect", "parameters": {"class_a": WATER, "class_b": URBAN}},
    ])

    result, trace = run(plan, GSD_10M, before=before, after=after)

    assert result["i1"] == pytest.approx(0.03)  # 3 px * 100 m2 / 10_000

    entry = trace.steps[0]
    assert entry.tool == "intersect"
    assert entry.task == "execute:i1"
    assert entry.output == pytest.approx(0.03)  # scalar output passes through the summary unchanged


def test_run_without_before_after_raises_when_plan_needs_intersect():
    plan = Plan.model_validate([
        {"id": "i1", "tool": "intersect", "parameters": {"class_a": WATER, "class_b": URBAN}},
    ])
    with pytest.raises(ExecutorError, match="before and after"):
        run(plan, GSD_10M)


# --- mixed plans, invalid plans, and replayability ---------------------------


def test_run_mixed_ops_and_change_plan():
    before, after = _before_after()
    plan = Plan.model_validate([
        {"id": "s1", "tool": "count", "parameters": {"class_id": WATER, "min_area_m2": 0}},
        {"id": "c1", "tool": "change", "parameters": {}},
    ])

    result, trace = run(plan, GSD_10M, mask=_water_mask(), before=before, after=after)

    assert result["s1"] == 1
    assert isinstance(result["c1"], ChangeReport)
    assert len(trace) == 2


def test_run_rejects_invalid_plan_before_touching_any_tool():
    plan = Plan.model_validate([{"id": "s1", "tool": "levitate", "parameters": {}}])
    with pytest.raises(PlanValidationError):
        run(plan, GSD_10M, mask=_water_mask())


def test_run_rejects_tool_with_no_backing_implementation():
    # 'segment' is a registered tool (agent/registry.py) but this executor
    # only backs count/size/presence/adjacency/change/intersect/caption/ground.
    plan = Plan.model_validate([{"id": "s1", "tool": "segment", "parameters": {}}])
    with pytest.raises(PlanValidationError, match="no implementation"):
        run(plan, GSD_10M, mask=_water_mask())


def test_run_executes_caption_tool(monkeypatch):
    from tools import caption as caption_tool
    monkeypatch.setattr(caption_tool, "GOOGLE_API_KEY", None)  # no client given -- must not hit the real API

    plan = Plan.model_validate([{"id": "cap1", "tool": "caption", "parameters": {}}])
    result, trace = run(plan, GSD_10M, mask=_water_mask())

    assert isinstance(result["cap1"], CaptionResult)
    assert result["cap1"].smoothed is False  # no Gemini available -- template used verbatim
    assert result["cap1"].facts  # the mask's real classes were extracted

    entry = trace.steps[0]
    assert entry.tool == "caption"
    assert entry.task == "execute:cap1"
    # the trace holds a SUMMARY (plain dict), not the raw CaptionResult dataclass
    assert entry.output["caption"] == result["cap1"].caption
    assert entry.output["template_sentence"] == result["cap1"].template_sentence
    assert entry.output["smoothed"] is False
    assert isinstance(entry.output["facts"], list)


def test_run_caption_respects_max_length_parameter(monkeypatch):
    from tools import caption as caption_tool
    monkeypatch.setattr(caption_tool, "GOOGLE_API_KEY", None)

    plan = Plan.model_validate([{"id": "cap1", "tool": "caption", "parameters": {"max_length": 5}}])
    result, _trace = run(plan, GSD_10M, mask=_water_mask())

    assert len(result["cap1"].caption) == 5


def test_run_without_mask_raises_when_plan_needs_caption():
    plan = Plan.model_validate([{"id": "cap1", "tool": "caption", "parameters": {}}])
    with pytest.raises(ExecutorError, match="caption"):
        run(plan, GSD_10M)


# --- automatic verification of every caption step's output -----------------


def test_run_automatically_verifies_caption_output(monkeypatch):
    # Per CLAUDE.md's "No pixel, no claim": a caption step must always be
    # independently re-checked, whether or not the plan itself asked for
    # verification -- this is the point of wiring it into the executor.
    from tools import caption as caption_tool
    monkeypatch.setattr(caption_tool, "GOOGLE_API_KEY", None)

    plan = Plan.model_validate([{"id": "cap1", "tool": "caption", "parameters": {}}])
    result, trace = run(plan, GSD_10M, mask=_water_mask())

    verify_entries = [e for e in trace if e.tool == "verify"]
    assert len(verify_entries) == 1
    entry = verify_entries[0]
    assert entry.task == "verify:cap1"
    assert entry.confidence == 1.0
    assert entry.parameters == {"answer": result["cap1"].caption}
    # no Gemini available in this test -- the template is used verbatim, so
    # every one of its own claims must recompute correctly against the same mask.
    assert entry.output["all_passed"] is True
    assert entry.output["claims"]  # the mask had real classes, so real claims were found
    assert entry.output["verified_answer"] == result["cap1"].caption


def test_run_does_not_verify_when_no_caption_step_ran():
    _, trace = run(_ops_plan(), GSD_10M, mask=_water_mask())
    assert not any(e.tool == "verify" for e in trace)


# --- the 'verify' tool, explicitly planned --------------------------------------


def test_run_executes_verify_tool_explicitly():
    from tools.verifier import VerificationResult

    # SEGMENTATION_CLASSES[WATER] == "Industrial or commercial units" -- the
    # real class name, not this test file's own WATER=1 alias, is what
    # tools/verifier.py resolves against.
    plan = Plan.model_validate([
        {"id": "v1", "tool": "verify", "parameters": {"answer": "0.04 ha of Industrial or commercial units"}},
    ])
    result, trace = run(plan, GSD_10M, mask=_water_mask())

    assert isinstance(result["v1"], VerificationResult)
    entry = trace.steps[0]
    assert entry.task == "execute:v1"
    assert entry.tool == "verify"
    assert entry.output["all_passed"] is True


def test_run_without_mask_raises_when_plan_needs_verify():
    plan = Plan.model_validate([{"id": "v1", "tool": "verify", "parameters": {"answer": "x"}}])
    with pytest.raises(ExecutorError, match="verify"):
        run(plan, GSD_10M)


def test_run_executes_ground_tool():
    # _water_mask()'s WATER(1) is real class id 1 = "Industrial or
    # commercial units" in agent.vocabulary.SEGMENTATION_CLASSES -- query
    # for that, not literal "water".
    plan = Plan.model_validate([{"id": "g1", "tool": "ground", "parameters": {"query": "the industrial units"}}])
    result, trace = run(plan, GSD_10M, mask=_water_mask())

    assert isinstance(result["g1"], GroundingResult)
    assert result["g1"].class_name == "Industrial or commercial units"
    assert result["g1"].pixel_bbox == (1, 1, 2, 2)

    entry = trace.steps[0]
    assert entry.tool == "ground"
    assert entry.task == "execute:g1"
    # the trace holds a SUMMARY (plain dict), not the raw GroundingResult dataclass
    assert entry.output["class_name"] == "Industrial or commercial units"
    assert entry.output["pixel_bbox"] == (1, 1, 2, 2)
    assert entry.output["pixel_polygon"]["type"] == "Polygon"
    assert entry.output["geo_bbox"] is None  # GSD_10M carries no transform/crs


def test_run_without_mask_raises_when_plan_needs_ground():
    plan = Plan.model_validate([{"id": "g1", "tool": "ground", "parameters": {"query": "the water body"}}])
    with pytest.raises(ExecutorError, match="ground"):
        run(plan, GSD_10M)


def test_run_executes_cross_modal_tool():
    # cross_modal segments internally (optical/sar/fused), so it needs the
    # raw 16-channel stack, not a pre-computed mask -- uses the real model
    # on a small synthetic stack (no fake session hook exists at the
    # executor layer, unlike tools/test_cross_modal.py's unit tests).
    rng = np.random.default_rng(0)
    stack = (rng.random((16, 120, 120)) * 1000).astype(np.float32)

    plan = Plan.model_validate([{"id": "x1", "tool": "cross_modal", "parameters": {}}])
    result, trace = run(plan, GSD_10M, stack=stack)

    assert isinstance(result["x1"], CrossModalResult)

    entry = trace.steps[0]
    assert entry.tool == "cross_modal"
    assert entry.task == "execute:x1"
    # the trace holds a SUMMARY (plain dict), not the raw CrossModalResult dataclass
    assert entry.output["cloud_simulated"] is False
    assert entry.output["summary"] == result["x1"].summary
    assert isinstance(entry.output["findings"], list)


def test_run_cross_modal_respects_cloud_simulation_parameter():
    rng = np.random.default_rng(0)
    stack = (rng.random((16, 120, 120)) * 1000).astype(np.float32)

    plan = Plan.model_validate([{"id": "x1", "tool": "cross_modal", "parameters": {"cloud_simulation": True}}])
    result, _trace = run(plan, GSD_10M, stack=stack)

    assert result["x1"].cloud_simulated is True
    assert result["x1"].cloud_fraction == pytest.approx(0.6)


def test_run_without_stack_raises_when_plan_needs_cross_modal():
    plan = Plan.model_validate([{"id": "x1", "tool": "cross_modal", "parameters": {}}])
    with pytest.raises(ExecutorError, match="cross_modal"):
        run(plan, GSD_10M, mask=_water_mask())  # a mask alone isn't enough -- cross_modal needs the raw stack


def test_run_executes_fusion_tool():
    # fusion needs the raw stack too (real optical bands for its spectral
    # index, real SAR channels when present) -- real model, no fake
    # session hook at the executor layer, same as cross_modal above.
    rng = np.random.default_rng(0)
    stack = (rng.random((16, 120, 120)) * 1000).astype(np.float32)

    plan = Plan.model_validate([{"id": "f1", "tool": "fusion", "parameters": {"class_id": 1}}])
    result, trace = run(plan, GSD_10M, stack=stack)

    assert isinstance(result["f1"], FusionResult)

    entry = trace.steps[0]
    assert entry.tool == "fusion"
    assert entry.task == "execute:f1"
    # the trace holds a SUMMARY (plain dict), not the raw FusionResult dataclass
    assert entry.output["class_id"] == 1
    assert entry.output["summary"] == result["f1"].summary
    assert isinstance(entry.output["sources"], list) and len(entry.output["sources"]) == 4
    assert set(entry.output["agreement"]) == {"segmentation", "classification", "spectral_index", "sar"}
    assert "encoder" in entry.output["weighting_note"]


def test_run_without_stack_raises_when_plan_needs_fusion():
    plan = Plan.model_validate([{"id": "f1", "tool": "fusion", "parameters": {"class_id": 1}}])
    with pytest.raises(ExecutorError, match="fusion"):
        run(plan, GSD_10M, mask=_water_mask())  # a mask alone isn't enough -- fusion needs the raw stack


def test_run_executes_metadata_tool():
    metadata = {"gsd_metres": 10.0, "filename": "S2A_MSIL2A_20170617T113321_N9999_R080_T29UPU_13_55.npz", "sensor": "fused"}
    plan = Plan.model_validate([{"id": "m1", "tool": "metadata", "parameters": {"aspect": "resolution"}}])
    result, trace = run(plan, metadata, mask=_water_mask())

    assert isinstance(result["m1"], MetadataResult)
    assert result["m1"].aspect == "resolution"

    entry = trace.steps[0]
    assert entry.tool == "metadata"
    assert entry.task == "execute:m1"
    assert entry.confidence == pytest.approx(1.0)  # metadata is fact retrieval -- always full confidence
    # the trace holds a SUMMARY (plain dict), not the raw MetadataResult dataclass
    assert entry.output["aspect"] == "resolution"
    assert entry.output["answer_text"] == result["m1"].answer_text


def test_run_metadata_date_aspect_reads_the_real_filename():
    metadata = {"gsd_metres": 10.0, "filename": "S2A_MSIL2A_20170617T113321_N9999_R080_T29UPU_13_55.npz", "sensor": "fused"}
    plan = Plan.model_validate([{"id": "m1", "tool": "metadata", "parameters": {"aspect": "date"}}])
    result, _trace = run(plan, metadata, mask=_water_mask())
    assert result["m1"].detail["acquisition_date"] == "2017-06-17"


def test_run_without_mask_raises_when_plan_needs_metadata():
    plan = Plan.model_validate([{"id": "m1", "tool": "metadata", "parameters": {"aspect": "date"}}])
    with pytest.raises(ExecutorError, match="metadata"):
        run(plan, GSD_10M)


def test_run_executes_conversational_tool_with_no_raster_at_all():
    plan = Plan.model_validate([{"id": "c1", "tool": "conversational", "parameters": {"kind": "greeting"}}])
    result, trace = run(plan, GSD_10M)  # no mask, no stack, no before/after -- none needed

    assert isinstance(result["c1"], ConversationalReply)
    assert result["c1"].kind == "greeting"

    entry = trace.steps[0]
    assert entry.tool == "conversational"
    assert entry.confidence == pytest.approx(1.0)
    assert entry.output["kind"] == "greeting"
    assert entry.output["reply"] == result["c1"].reply


def test_run_conversational_off_topic_kind():
    plan = Plan.model_validate([{"id": "c1", "tool": "conversational", "parameters": {"kind": "off_topic"}}])
    result, _trace = run(plan, GSD_10M)
    assert result["c1"].kind == "off_topic"


def test_run_is_replayable_without_the_llm():
    mask = _water_mask()
    plan = Plan.model_validate_json(_ops_plan().model_dump_json())  # simulate loading from storage

    result_1, trace_1 = run(plan, GSD_10M, mask=mask)
    result_2, trace_2 = run(plan, GSD_10M, mask=mask)

    assert result_1 == result_2
    assert [e.model_dump() for e in trace_1] == [e.model_dump() for e in trace_2]


# --- confidence/engine.py wiring ----------------------------------------------
# Gated on stack+session+config ALL being given (see run()'s own docstring):
# without them, every step keeps dsl.execute()'s own exact 1.0 unchanged --
# already covered by every test above, none of which pass those three.


@dataclass
class _FakeInferenceResult:
    probabilities: np.ndarray
    mask: np.ndarray


def _fake_session_config_stack():
    # Real class objects aren't needed -- executor.run() only ever passes
    # these straight through to perception.infer.segment_image, which is
    # monkeypatched in every test below, so they're pure sentinels here.
    return object(), object(), np.zeros((16, 10, 10), dtype=np.float32)


def _patch_segment_image(monkeypatch, probabilities: np.ndarray):
    # tta_stability() calls this 8 more times (once per D4 transform) on
    # top of the one call run() itself makes for the perception margin --
    # every call gets the SAME probabilities/mask here regardless of which
    # transform was requested, which is fine: a fake segment_image doesn't
    # need to be transform-aware for tests that only check the WIRING
    # (does the confidence engine get called, does its output reach the
    # trace) rather than the D4 geometry itself (covered separately, and
    # exactly, by test_confidence_engine.py's own D4 round-trip tests).
    mask = np.argmax(probabilities, axis=0)
    monkeypatch.setattr(
        "perception.infer.segment_image",
        lambda stack, sensor, *, session, config: _FakeInferenceResult(probabilities=probabilities, mask=mask),
    )


def _fake_confidence_result(calibrated: float = 0.61) -> ConfidenceResult:
    return ConfidenceResult(
        calibrated=calibrated,
        components={"perception_margin": calibrated, "tta_stability": None,
                    "cross_source_agreement": None, "resolution_suitability": 1.0, "plan_validity": None},
        weights_used={"perception_margin": 0.667, "resolution_suitability": 0.333},
        refused=calibrated < 0.55,
    )


def test_run_overrides_confidence_when_stack_session_config_all_given(monkeypatch):
    session, config, stack = _fake_session_config_stack()
    probabilities = np.zeros((19, 10, 10), dtype=np.float32)
    probabilities[WATER] = 1.0
    _patch_segment_image(monkeypatch, probabilities)
    monkeypatch.setattr(executor_module.confidence_engine, "compute_confidence", lambda inputs: _fake_confidence_result(0.61))

    plan = Plan.model_validate([{"id": "s1", "tool": "presence", "parameters": {"class_id": WATER, "min_area_m2": 0}}])
    _, trace = run(plan, GSD_10M, mask=_water_mask(), stack=stack, session=session, config=config)

    execute_step = next(e for e in trace if e.task == "execute:s1")
    assert execute_step.confidence == pytest.approx(0.61)  # no longer the hardcoded 1.0


def test_run_appends_a_confidence_trace_step_with_the_full_breakdown(monkeypatch):
    session, config, stack = _fake_session_config_stack()
    _patch_segment_image(monkeypatch, np.full((19, 10, 10), 1.0 / 19, dtype=np.float32))
    fake_result = _fake_confidence_result(0.42)
    monkeypatch.setattr(executor_module.confidence_engine, "compute_confidence", lambda inputs: fake_result)

    plan = Plan.model_validate([{"id": "s1", "tool": "presence", "parameters": {"class_id": WATER, "min_area_m2": 0}}])
    _, trace = run(plan, GSD_10M, mask=_water_mask(), stack=stack, session=session, config=config)

    confidence_step = next(e for e in trace if e.task == "confidence:s1")
    assert confidence_step.tool == "confidence.engine.compute_confidence"
    assert confidence_step.confidence == pytest.approx(0.42)
    assert confidence_step.output["calibrated"] == pytest.approx(0.42)
    assert confidence_step.output["components"] == fake_result.components
    assert confidence_step.output["geometry_confidence"] == 1.0  # always, per DETERMINISTIC_GEOMETRY_CONFIDENCE
    assert confidence_step.output["refused"] is True  # 0.42 < REFUSAL_THRESHOLD


def test_run_confidence_wiring_leaves_non_qualifying_tools_untouched(monkeypatch):
    session, config, stack = _fake_session_config_stack()
    _patch_segment_image(monkeypatch, np.full((19, 10, 10), 1.0 / 19, dtype=np.float32))
    monkeypatch.setattr(executor_module.confidence_engine, "compute_confidence", lambda inputs: _fake_confidence_result())

    plan = Plan.model_validate([{"id": "cap1", "tool": "caption", "parameters": {}}])
    _, trace = run(plan, GSD_10M, mask=_water_mask(), stack=stack, session=session, config=config)

    execute_step = next(e for e in trace if e.task == "execute:cap1")
    assert execute_step.confidence == pytest.approx(1.0)  # caption isn't a confidence-qualifying tool
    assert not any(e.task == "confidence:cap1" for e in trace)


def test_run_adjacency_confidence_uses_the_union_of_both_classes(monkeypatch):
    session, config, stack = _fake_session_config_stack()
    _patch_segment_image(monkeypatch, np.full((19, 10, 10), 1.0 / 19, dtype=np.float32))
    captured_inputs = []

    def fake_compute_confidence(inputs):
        captured_inputs.append(inputs)
        return _fake_confidence_result()

    monkeypatch.setattr(executor_module.confidence_engine, "compute_confidence", fake_compute_confidence)

    plan = Plan.model_validate([
        {"id": "a1", "tool": "adjacency", "parameters": {"class_a": WATER, "class_b": URBAN, "distance_m": 0}},
    ])
    run(plan, GSD_10M, mask=_water_mask(), stack=stack, session=session, config=config)

    assert captured_inputs[0].class_id == [WATER, URBAN]


def test_run_without_stack_session_or_config_never_calls_the_confidence_engine(monkeypatch):
    called = []
    monkeypatch.setattr(executor_module.confidence_engine, "compute_confidence", lambda inputs: called.append(1))

    result, trace = run(_ops_plan(), GSD_10M, mask=_water_mask())  # no stack/session/config

    assert called == []
    entry = next(e for e in trace if e.task == "execute:s1")
    assert entry.confidence == pytest.approx(1.0)  # unchanged, exactly today's existing behaviour


def test_run_confidence_wiring_passes_plan_attempts_through(monkeypatch):
    # Uses the REAL confidence_engine.compute_confidence (not mocked), to
    # prove plan_attempts genuinely flows all the way through -- so
    # evidence.fusion.fuse_evidence (which needs a real ClassConfig for its
    # classification-head lookup, not the bare sentinel _fake_session_config_
    # stack() returns) is stubbed out instead, the one real-model call this
    # particular test doesn't care about.
    session, config, stack = _fake_session_config_stack()
    _patch_segment_image(monkeypatch, np.full((19, 10, 10), 1.0 / 19, dtype=np.float32))

    @dataclass
    class _FakeFusionResult:
        agreement_fraction: float

    monkeypatch.setattr("evidence.fusion.fuse_evidence", lambda *a, **k: _FakeFusionResult(agreement_fraction=0.5))

    plan = Plan.model_validate([{"id": "s1", "tool": "size", "parameters": {"class_id": WATER}}])
    _, trace = run(
        plan, GSD_10M, mask=_water_mask(), stack=stack, session=session, config=config, plan_attempts=3,
    )

    confidence_step = next(e for e in trace if e.task == "confidence:s1")
    assert confidence_step.output["components"]["plan_validity"] == pytest.approx(plan_validity(3))


# --- real model, end to end (no monkeypatching) -------------------------------
# Mirrors tests/test_infer.py's own module-scoped real-session fixture --
# a genuine, if slow-ish (~1-2s per test), integration check that the full
# wiring (re-derived probabilities, 8-way TTA, evidence/fusion.py
# agreement, all against the REAL model) doesn't crash end to end and
# produces a sane, in-range calibrated number -- exact values aren't
# asserted since real model output on a real patch isn't hand-computable.


@pytest.fixture(scope="module")
def real_session():
    from perception import infer
    return infer.load_model()


@pytest.fixture(scope="module")
def real_config():
    from perception import infer
    return infer.load_class_config()


@pytest.fixture(scope="module")
def real_stack() -> np.ndarray:
    patches_dir = Path(__file__).resolve().parent.parent / "data" / "demo_patches"
    npz_path = sorted(patches_dir.glob("*.npz"))[0]
    data = np.load(npz_path, allow_pickle=True)
    return data["stack"][:16].astype(np.float32)


def test_run_confidence_wiring_end_to_end_against_the_real_model(real_session, real_config, real_stack):
    from perception import infer

    real_mask = infer.segment_image(real_stack, "fused", session=real_session, config=real_config).mask
    present_class = int(np.bincount(real_mask.flatten()).argmax())  # whatever's actually in this patch

    classes = dict(enumerate(SEGMENTATION_CLASSES))
    plan = Plan.model_validate([{"id": "s1", "tool": "presence", "parameters": {"class_id": present_class, "min_area_m2": 0}}])

    _, trace = run(
        plan, {"gsd_metres": 10.0}, mask=real_mask, stack=real_stack,
        session=real_session, config=real_config, classes=classes, plan_attempts=1,
    )

    execute_step = next(e for e in trace if e.task == "execute:s1")
    confidence_step = next(e for e in trace if e.task == "confidence:s1")

    assert 0.0 <= execute_step.confidence <= 1.0
    assert execute_step.confidence == pytest.approx(confidence_step.output["calibrated"])
    assert confidence_step.output["geometry_confidence"] == 1.0
    assert confidence_step.output["components"]["perception_margin"] is not None
    assert confidence_step.output["components"]["tta_stability"] is not None
    assert confidence_step.output["components"]["cross_source_agreement"] is not None
    assert confidence_step.output["components"]["plan_validity"] == pytest.approx(1.0)
