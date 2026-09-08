"""Tests for agent/dsl.py: the typed plan DSL -- validation, JSON
serialisation/replay, and execution against real tool implementations
without any LLM involved."""

import functools

import numpy as np
import pydantic
import pytest

from agent.dsl import ExecutionResult, Plan, PlanValidationError, Step, execute, validate
from evidence import ops

WATER = 1
LAND = 0
GSD_10M = {"gsd_metres": 10.0}


def _water_mask() -> np.ndarray:
    mask = np.full((10, 10), LAND, dtype=int)
    mask[1:3, 1:3] = WATER  # a single 2x2 (400 m2) water blob
    return mask


def _valid_plan_dicts() -> list[dict]:
    return [
        {"id": "s1", "tool": "count", "parameters": {"class_id": WATER, "min_area_m2": 0}},
        {"id": "s2", "tool": "presence", "parameters": {"class_id": WATER, "min_area_m2": 1000}},
        {"id": "s3", "tool": "size", "parameters": {"class_id": WATER}},
    ]


# --- Plan: typed list, serialisation -----------------------------------------


def test_plan_parses_from_list_of_dicts():
    plan = Plan.model_validate(_valid_plan_dicts())
    assert len(plan) == 3
    assert [step.id for step in plan] == ["s1", "s2", "s3"]
    assert plan[0].tool == "count"


def test_step_rejects_unknown_field():
    with pytest.raises(pydantic.ValidationError):
        Step(id="s1", tool="count", parameters={}, unexpected="nope")


def test_plan_round_trips_through_json():
    plan = Plan.model_validate(_valid_plan_dicts())
    json_text = plan.model_dump_json()

    replayed = Plan.model_validate_json(json_text)

    assert replayed == plan
    assert [s.model_dump() for s in replayed] == [s.model_dump() for s in plan]


# --- validate(): rejects unknown tools and out-of-schema parameters ---------


def test_validate_accepts_a_well_formed_plan():
    validate(Plan.model_validate(_valid_plan_dicts()))  # no raise


def test_validate_rejects_unknown_tool():
    plan = Plan.model_validate([{"id": "s1", "tool": "levitate", "parameters": {}}])
    with pytest.raises(PlanValidationError, match="unknown tool"):
        validate(plan)


def test_validate_rejects_parameter_outside_permitted_schema():
    plan = Plan.model_validate(
        [{"id": "s1", "tool": "count", "parameters": {"class_id": 1, "bogus": True}}]
    )
    with pytest.raises(PlanValidationError, match="s1"):
        validate(plan)


def test_validate_rejects_wrong_parameter_type():
    plan = Plan.model_validate(
        [{"id": "s1", "tool": "count", "parameters": {"class_id": "one", "min_area_m2": 0}}]
    )
    with pytest.raises(PlanValidationError):
        validate(plan)


def test_validate_rejects_duplicate_step_ids():
    plan = Plan.model_validate([
        {"id": "s1", "tool": "count", "parameters": {"class_id": 1, "min_area_m2": 0}},
        {"id": "s1", "tool": "size", "parameters": {"class_id": 1}},
    ])
    with pytest.raises(PlanValidationError, match="duplicate"):
        validate(plan)


# --- execute(): replayable, without the LLM ---------------------------------


def test_execute_replays_real_evidence_ops_without_llm():
    mask = _water_mask()
    tool_functions = {
        "count": functools.partial(ops.count, mask, metadata=GSD_10M),
        "presence": functools.partial(ops.presence, mask, metadata=GSD_10M),
        "size": functools.partial(ops.size, mask, metadata=GSD_10M),
    }

    plan = Plan.model_validate(_valid_plan_dicts())
    result = execute(plan, tool_functions)

    assert isinstance(result, ExecutionResult)
    assert result.outputs["s1"] == 1  # one blob, no area filter
    assert result.outputs["s2"] is False  # blob is 400 m2, threshold is 1000 m2
    assert result.outputs["s3"] == pytest.approx(0.04)  # 4 px * 100 m2 / 10_000 = 0.04 ha


def test_execute_is_deterministic_across_replays():
    mask = _water_mask()
    tool_functions = {
        "count": functools.partial(ops.count, mask, metadata=GSD_10M),
        "presence": functools.partial(ops.presence, mask, metadata=GSD_10M),
        "size": functools.partial(ops.size, mask, metadata=GSD_10M),
    }
    plan = Plan.model_validate_json(Plan.model_validate(_valid_plan_dicts()).model_dump_json())

    first = execute(plan, tool_functions)
    second = execute(plan, tool_functions)

    assert first.outputs == second.outputs


def test_execute_produces_a_trace_entry_per_step():
    mask = _water_mask()
    tool_functions = {"count": functools.partial(ops.count, mask, metadata=GSD_10M)}
    plan = Plan.model_validate(
        [{"id": "s1", "tool": "count", "parameters": {"class_id": WATER, "min_area_m2": 0}}]
    )

    result = execute(plan, tool_functions)

    assert len(result.trace) == 1
    entry = result.trace[0]
    assert entry.tool == "count"
    assert entry.task == "execute:s1"
    assert entry.parameters == {"class_id": WATER, "min_area_m2": 0}
    assert entry.output == 1
    assert entry.confidence == 1.0


def test_execute_validates_before_running_anything():
    calls = []
    tool_functions = {"count": lambda **kwargs: calls.append(kwargs)}
    plan = Plan.model_validate([{"id": "s1", "tool": "levitate", "parameters": {}}])

    with pytest.raises(PlanValidationError):
        execute(plan, tool_functions)
    assert calls == []  # nothing was invoked -- validation failed first


def test_execute_raises_when_no_implementation_provided():
    plan = Plan.model_validate(
        [{"id": "s1", "tool": "count", "parameters": {"class_id": WATER, "min_area_m2": 0}}]
    )
    with pytest.raises(PlanValidationError, match="no implementation"):
        execute(plan, tool_functions={})
