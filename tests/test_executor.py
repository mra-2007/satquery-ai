"""Tests for agent/executor.py: running a validated Plan against the real
evidence/ops.py and evidence/change.py implementations, and the resulting
ExecutionTrace's replayability."""

import numpy as np
import pytest

from agent.dsl import Plan, PlanValidationError
from agent.executor import ExecutionTrace, ExecutorError, run
from evidence.change import ChangeReport
from tools.caption import CaptionResult
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
    }


def test_run_without_before_after_raises_when_plan_needs_change():
    plan = Plan.model_validate([{"id": "c1", "tool": "change", "parameters": {}}])
    with pytest.raises(ExecutorError, match="before and after"):
        run(plan, GSD_10M)


def test_run_with_only_before_raises():
    before, _after = _before_after()
    plan = Plan.model_validate([{"id": "c1", "tool": "change", "parameters": {}}])
    with pytest.raises(ExecutorError):
        run(plan, GSD_10M, before=before)


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
    # only backs count/size/presence/adjacency/change/caption/ground.
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


def test_run_is_replayable_without_the_llm():
    mask = _water_mask()
    plan = Plan.model_validate_json(_ops_plan().model_dump_json())  # simulate loading from storage

    result_1, trace_1 = run(plan, GSD_10M, mask=mask)
    result_2, trace_2 = run(plan, GSD_10M, mask=mask)

    assert result_1 == result_2
    assert [e.model_dump() for e in trace_1] == [e.model_dump() for e in trace_2]
