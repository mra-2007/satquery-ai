"""Tests for agent/guardrail.py: rewriting an unresolvable 'count' step to
'size', driven fresh from the image's GSD each call -- never hardcoded."""

import pytest

from agent.dsl import Plan, validate
from agent.guardrail import apply_capability_guardrail

GSD_10M = 10.0  # min_resolvable = 25 m
GSD_2M = 2.0    # min_resolvable = 5 m


def _plan(*steps: dict) -> Plan:
    return Plan.model_validate(list(steps))


def test_uncountable_class_is_rewritten_to_size():
    plan = _plan({"id": "s1", "tool": "count", "parameters": {"class_id": 3, "min_area_m2": 0}})
    classes = {3: "building"}  # 10 m typical size <= 25 m min resolvable at 10 m GSD

    rewritten, trace = apply_capability_guardrail(plan, classes, GSD_10M)

    assert rewritten[0].tool == "size"
    assert rewritten[0].parameters == {"class_id": 3}
    assert len(trace) == 1
    entry = trace[0]
    assert entry.task == "capability_guardrail"
    assert entry.tool == "agent.guardrail.apply_capability_guardrail"
    assert entry.parameters == {"original_tool": "count", "class_id": 3, "class_name": "building"}
    assert entry.output["degraded_to"] == "size"
    assert "building" in entry.output["limitation"]
    assert entry.confidence == 1.0


def test_countable_class_is_left_unchanged():
    plan = _plan({"id": "s1", "tool": "count", "parameters": {"class_id": 1, "min_area_m2": 0}})
    classes = {1: "water_body"}  # 30 m typical size > 25 m min resolvable at 10 m GSD

    rewritten, trace = apply_capability_guardrail(plan, classes, GSD_10M)

    assert rewritten[0].tool == "count"
    assert rewritten[0].parameters == {"class_id": 1, "min_area_m2": 0}
    assert trace == []


def test_non_count_steps_are_never_touched():
    plan = _plan(
        {"id": "s1", "tool": "size", "parameters": {"class_id": 3}},
        {"id": "s2", "tool": "presence", "parameters": {"class_id": 3, "min_area_m2": 0}},
    )
    classes = {3: "building"}

    rewritten, trace = apply_capability_guardrail(plan, classes, GSD_10M)

    assert [s.tool for s in rewritten] == ["size", "presence"]
    assert trace == []


def test_unknown_class_name_is_left_unchanged():
    plan = _plan({"id": "s1", "tool": "count", "parameters": {"class_id": 99, "min_area_m2": 0}})
    rewritten, trace = apply_capability_guardrail(plan, classes={}, gsd_metres=GSD_10M)

    assert rewritten[0].tool == "count"
    assert trace == []


def test_class_name_absent_from_capability_table_is_left_unchanged():
    plan = _plan({"id": "s1", "tool": "count", "parameters": {"class_id": 3, "min_area_m2": 0}})
    classes = {3: "urban"}  # not one of agent.registry.TYPICAL_OBJECT_SIZE_M's keys

    rewritten, trace = apply_capability_guardrail(plan, classes, GSD_10M)

    assert rewritten[0].tool == "count"
    assert trace == []


def test_only_the_targeted_step_is_rewritten_in_a_mixed_plan():
    plan = _plan(
        {"id": "s1", "tool": "count", "parameters": {"class_id": 1, "min_area_m2": 0}},   # water_body: stays
        {"id": "s2", "tool": "count", "parameters": {"class_id": 3, "min_area_m2": 0}},   # building: rewritten
    )
    classes = {1: "water_body", 3: "building"}

    rewritten, trace = apply_capability_guardrail(plan, classes, GSD_10M)

    assert [(s.id, s.tool) for s in rewritten] == [("s1", "count"), ("s2", "size")]
    assert len(trace) == 1
    assert trace[0].parameters["class_id"] == 3


def test_countability_is_recomputed_not_hardcoded_across_gsd():
    plan = _plan({"id": "s1", "tool": "count", "parameters": {"class_id": 3, "min_area_m2": 0}})
    classes = {3: "building"}

    coarse, coarse_trace = apply_capability_guardrail(plan, classes, GSD_10M)
    fine, fine_trace = apply_capability_guardrail(plan, classes, GSD_2M)

    assert coarse[0].tool == "size"   # not countable at 10 m
    assert len(coarse_trace) == 1
    assert fine[0].tool == "count"    # countable at 2 m (10 m > 5 m min resolvable)
    assert fine_trace == []


def test_rewritten_plan_is_itself_valid():
    plan = _plan({"id": "s1", "tool": "count", "parameters": {"class_id": 3, "min_area_m2": 0}})
    rewritten, _trace = apply_capability_guardrail(plan, {3: "building"}, GSD_10M)
    validate(rewritten)  # no raise
