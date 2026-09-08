"""The plan DSL: the "typed, replayable tool graph" from the deck.

The LLM's only job is to emit a `Plan` -- an ordered, typed list of tool
calls. Once emitted, the plan is a plain data structure: `validate()`
checks it against the tool registry (agent/registry.py) with no model in
the loop, `execute()` runs it against real tool implementations with no
model in the loop, and because it's a pydantic model it serialises to and
from JSON losslessly, so a plan can be logged, stored, handed to a
different process, and replayed byte-for-byte identically later. That
replayability is the point: the LLM is asked once, and the resulting plan
is then deterministic and auditable, per CLAUDE.md's "No pixel, no claim."

    from agent.dsl import Plan, validate, execute

    plan = Plan.model_validate([
        {"id": "s1", "tool": "count", "parameters": {"class_id": 1, "min_area_m2": 500}},
    ])
    validate(plan)                        # raises PlanValidationError if not
    json_text = plan.model_dump_json()    # serialise
    replayed = Plan.model_validate_json(json_text)  # deserialise, elsewhere/later
    result = execute(replayed, tool_functions={"count": my_count_fn})
"""

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, RootModel

from agent.registry import REGISTRY, ParameterValidationError, validate_parameters
from evidence.schema import TraceStep


class PlanValidationError(ValueError):
    """Raised by validate() (and by execute(), which validates first) when a
    plan references an unknown tool, has out-of-schema parameters, or
    duplicate step ids."""


class Step(BaseModel):
    """One tool call in a plan. `id` makes each step's output addressable
    (execute() keys its results by it) and lets validate() catch duplicate
    or malformed plans."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Unique id for this step within the plan.")
    tool: str = Field(..., description="Name of a tool in agent.registry.REGISTRY.")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Keyword arguments for the tool call."
    )


class Plan(RootModel[list[Step]]):
    """A typed, ordered list of steps -- serialise with
    `.model_dump_json()` / `.model_dump()`, deserialise with
    `Plan.model_validate_json(...)` / `Plan.model_validate(...)`."""

    def __iter__(self):
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)

    def __getitem__(self, index: int) -> Step:
        return self.root[index]


def validate(plan: Plan) -> None:
    """Validate every step against the tool registry: unknown tool names,
    out-of-schema parameters, and duplicate step ids are all rejected.
    Raises PlanValidationError; returns None on success."""
    seen_ids: set[str] = set()

    for step in plan:
        if step.id in seen_ids:
            raise PlanValidationError(f"duplicate step id {step.id!r}")
        seen_ids.add(step.id)

        tool = REGISTRY.get(step.tool)
        if tool is None:
            raise PlanValidationError(
                f"step {step.id!r}: unknown tool {step.tool!r}; "
                f"registered tools: {sorted(REGISTRY)}"
            )
        try:
            validate_parameters(tool, step.parameters)
        except ParameterValidationError as exc:
            raise PlanValidationError(f"step {step.id!r}: {exc}") from exc


@dataclass
class ExecutionResult:
    outputs: dict[str, Any]  # step.id -> tool output
    trace: list[TraceStep]


def execute(plan: Plan, tool_functions: dict[str, Callable[..., Any]]) -> ExecutionResult:
    """Replay a plan against real tool implementations, without the LLM.

    `tool_functions` maps each tool name used in the plan to a callable
    invoked as `fn(**step.parameters)`; the caller supplies these (e.g.
    evidence.ops.count, evidence.change.detect_change) since the registry
    only describes tool contracts, not implementations. Validates the plan
    first, so an invalid plan never partially executes. Steps run in list
    order and each step's raw output is recorded in the trace under its
    tool name -- this function makes no assumption about later steps
    depending on earlier ones.
    """
    validate(plan)

    outputs: dict[str, Any] = {}
    trace: list[TraceStep] = []

    for step in plan:
        fn = tool_functions.get(step.tool)
        if fn is None:
            raise PlanValidationError(
                f"step {step.id!r}: no implementation provided for tool {step.tool!r}"
            )
        output = fn(**step.parameters)
        outputs[step.id] = output
        trace.append(TraceStep(
            task=f"execute:{step.id}",
            tool=step.tool,
            parameters=step.parameters,
            output=output,
            confidence=1.0,
        ))

    return ExecutionResult(outputs=outputs, trace=trace)
