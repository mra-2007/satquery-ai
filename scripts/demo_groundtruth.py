"""End-to-end demo: classify_task -> planner -> validate -> executor.

Builds a synthetic 120x120 px, 10 m GSD ground-truth class raster (the
"Sentinel patch" shape from CLAUDE.md's Key numbers) with known water
bodies, a forest patch, and built-up regions, then runs a natural-language
question all the way through the real pipeline: task classification, plan
generation, plan validation, and execution against evidence/ops.py -- and
prints the answer plus the full execution trace.

The capability guardrail (agent/guardrail.py) that degrades an
unresolvable 'count' to 'size' lives inside agent/executor.py's run(), not
here -- pass it `classes` and every caller gets the protection, this
script included.

Runs in forced-offline mode (classify_task/plan_from_query never call
Gemini) so the output is fast and fully deterministic; the real API path
is already covered by the mocked tests in tests/test_planner.py and
tests/test_tasks.py. This also demonstrates CLAUDE.md's "Must work
offline" constraint directly.

Run with the venv's own interpreter:
    venv\\Scripts\\python.exe -m scripts.demo_groundtruth
"""

import numpy as np

from agent import planner, tasks
from agent.dsl import PlanValidationError, validate
from agent.executor import run as run_plan
from agent.planner import SceneDescriptor, plan_from_query
from agent.tasks import classify_task
from evidence.schema import TraceStep

# Force offline mode: skip Gemini entirely so this demo is fast and
# reproducible. (See module docstring.)
planner.GOOGLE_API_KEY = None
tasks.GOOGLE_API_KEY = None

LAND, WATER, FOREST, BUILDING = 0, 1, 2, 3
GSD_METRES = 10.0

SCENE = SceneDescriptor(
    layers=["class_raster"],
    classes={LAND: "land", WATER: "water", FOREST: "forest", BUILDING: "building"},
    sensor="sentinel-2",
    gsd_metres=GSD_METRES,
    bbox=(77.590, 12.970, 77.602, 12.982),
)
METADATA = {"gsd_metres": GSD_METRES}


def build_groundtruth_mask() -> np.ndarray:
    """A synthetic 120x120 class-ID raster: 3 separate water bodies (ponds),
    one 40x40 px forest patch (16 ha), and 2 built-up blocks -- one
    touching a pond (for the adjacency question), one far from any water."""
    mask = np.full((120, 120), LAND, dtype=int)

    mask[10:15, 10:15] = WATER   # pond 1 (5x5 = 25 px)
    mask[10:15, 40:45] = WATER   # pond 2 (5x5 = 25 px)
    mask[80:86, 90:96] = WATER   # pond 3 (6x6 = 36 px)

    mask[60:100, 10:50] = FOREST  # 40x40 = 1600 px = 16 ha at 10 m GSD

    mask[15:20, 10:15] = BUILDING     # touches pond 1 (rows 14/15 boundary)
    mask[100:105, 100:105] = BUILDING  # far from every water body

    return mask


def render_answer(trace: list[TraceStep]) -> str:
    """Built from the trace's own 'execute:*' entries (not the original
    Plan) so it reflects whatever actually ran, including any rewrite the
    capability guardrail made inside executor.run()."""
    parts = []
    limitations = []

    for entry in trace:
        if entry.task == "capability_guardrail":
            limitations.append(entry.output["limitation"])
            continue
        if not entry.task.startswith("execute:"):
            continue

        value = entry.output
        class_name = SCENE.classes.get(entry.parameters.get("class_id", -1))
        if entry.tool == "count":
            parts.append(f"{value} {class_name}(s)")
        elif entry.tool == "size":
            parts.append(f"{value:.2f} ha of {class_name}")
        elif entry.tool in ("presence", "adjacency"):
            parts.append("yes" if value else "no")
        else:
            parts.append(str(value))

    answer = "; ".join(parts)
    for limitation in limitations:
        answer += f"\n    LIMITATION: {limitation}"
    return answer


def answer_query(query: str, mask: np.ndarray) -> None:
    print(f"\n{'=' * 78}\nQ: {query}\n{'=' * 78}")
    full_trace: list[TraceStep] = []

    classification = classify_task(query)
    full_trace += classification.trace
    print(f"  1. classify_task -> {classification.label!r} (confidence {classification.confidence:.2f})")

    plan = plan_from_query(query, SCENE)
    print(f"  2. planner       -> {[(s.tool, s.parameters) for s in plan]}")

    try:
        validate(plan)
    except PlanValidationError as exc:
        print(f"  REJECTED before execution: {exc}")
        return
    print("  3. validate      -> ok")

    # classes=SCENE.classes is what lets executor.run() apply the
    # capability guardrail -- this script no longer implements it itself.
    result, exec_trace = run_plan(plan, METADATA, mask=mask, classes=SCENE.classes)
    full_trace += list(exec_trace)
    print("  4. executor      -> ran")

    print(f"\n  ANSWER: {render_answer(full_trace)}")

    print("\n  Execution trace:")
    for entry in full_trace:
        print(
            f"    - task={entry.task!r:32s} tool={entry.tool!r:45s} "
            f"params={entry.parameters} output={entry.output} confidence={entry.confidence}"
        )


def main() -> None:
    mask = build_groundtruth_mask()
    questions = [
        "How many water bodies are in this area?",
        "How much forest is there in hectares?",
        "Is there any built-up area near water?",
        "How many buildings are there?",
    ]
    for question in questions:
        answer_query(question, mask)


if __name__ == "__main__":
    main()
