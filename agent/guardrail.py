"""The capability guardrail: rewrites a Plan so it never asks the system to
count something the image's own resolution can't actually resolve.

Per CLAUDE.md: buildings (~10 m typical size) are not countable at 10 m
GSD (minimum resolvable ~25 m), while larger features like water bodies
and forest patches are. agent.registry.capability_table computes that
per-class, per-image verdict; this module is what ACTS on it -- any
`count` step targeting a class that isn't resolvable is rewritten to an
equivalent `size` step instead, with a plain-English limitation recorded
in the trace.

This is applied inside agent/executor.py's run(), so every caller that
executes a plan -- the API, a script, anything -- gets this protection
automatically. It doesn't have to be remembered and called separately.
"""

from agent.dsl import Plan
from agent.registry import capability_table
from evidence.schema import TraceStep


def apply_capability_guardrail(
    plan: Plan,
    classes: dict[int, str],
    gsd_metres: float,
) -> tuple[Plan, list[TraceStep]]:
    """Rewrite any `count` step whose class isn't resolvable at this GSD
    into an equivalent `size` step.

    `classes` maps the plan's class_ids to the scene's human-readable
    class names -- the same names agent.registry.TYPICAL_OBJECT_SIZE_M
    uses (e.g. "building", "water_body"). A class with no entry there, or
    that IS resolvable, is left untouched. Returns (possibly rewritten
    plan, trace) -- the trace is empty when nothing was rewritten.
    """
    table = capability_table({"gsd_metres": gsd_metres})
    trace: list[TraceStep] = []
    new_steps = []

    for step in plan:
        if step.tool == "count":
            class_id = step.parameters["class_id"]
            # A list class_id (agent.vocabulary.resolve_noun resolving a
            # generic noun like "forest" to several real classes at once)
            # has no single name to look up here, and none of
            # TYPICAL_OBJECT_SIZE_M's keys are multi-class groupings
            # anyway -- treated the same as an unrecognised class: left
            # untouched, not degraded.
            class_name = None if isinstance(class_id, list) else classes.get(class_id, "")
            capability = table.classes.get(class_name) if class_name is not None else None
            if capability is not None and not capability.countable:
                limitation = (
                    f"Cannot count individual {class_name}s at {gsd_metres:.0f} m GSD "
                    f"(typical size {capability.typical_object_size_m:.0f} m <= minimum "
                    f"resolvable {capability.min_resolvable_m:.0f} m); reporting total "
                    f"{class_name} area instead."
                )
                new_steps.append({"id": step.id, "tool": "size", "parameters": {"class_id": class_id}})
                trace.append(TraceStep(
                    task="capability_guardrail",
                    tool="agent.guardrail.apply_capability_guardrail",
                    parameters={"original_tool": "count", "class_id": class_id, "class_name": class_name},
                    output={"degraded_to": "size", "limitation": limitation},
                    confidence=1.0,
                ))
                continue
        new_steps.append(step.model_dump())

    return Plan.model_validate(new_steps), trace
