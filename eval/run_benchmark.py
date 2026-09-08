"""Scores SatQuery AI's full pipeline against the BigEarthNet.txt benchmark
sample in data/bench_patches/: questions.csv (969 questions, patch_id ->
input/output/type/category) plus one <patch_id>.npz per referenced patch
(a real 16-channel reBEN Sentinel-1+2 stack and its own pixel-level
ground-truth mask).

This is a 969-question, 6% sample of the full 15,029-question
BigEarthNet.txt benchmark (data/benchmark/bigearthnet_txt_benchmark.csv),
limited to the 64 patches actually available locally as
data/bench_patches/*.npz -- see data/README.md and CLAUDE.md on why the
full reBEN corpus isn't stored locally. Treat every accuracy number below
as an estimate over that small sample, not the full benchmark's score.

Per question: run inference on the real stack (sensor_id="fused" -- these
are genuine Sentinel-1+2 stacks, like data/demo_patches/*.npz, not RGB-only
like RSVQA-LR) to get a predicted mask, then classify_task -> planner ->
validate -> executor, and compare the computed answer to the expected one.
Scored TWICE per question, against two different masks:
  - the PREDICTED mask (real model output) -- this is what the deployed
    system would actually answer with.
  - the GROUND-TRUTH mask (npz's own "mask" array) -- this holds the
    reasoning layer (classify_task/planner/validate/executor) fixed and
    perfect-information, isolating how much of the error comes from
    segmentation mistakes vs. how much comes from the reasoning layer
    itself. The gap between the two tables IS the model's segmentation
    error; whatever's left in the ground-truth table is the reasoning
    layer's own ceiling.

Only 4 of questions.csv's 10 categories are in scope, per the request:
presence, count, area (evidence/ops.py's "size" tool), and adjacency --
654 of the 969 questions. The other 315 (reference, relative pos, point,
season, climate zone, country) ask for grounding, captioning, or metadata
lookup that no evidence/ops.py tool answers at all; they are excluded
from the table entirely, not scored as unsupported.

Full 19-class vocabulary (no 4-bucket collapse)
--------------------------------------------------
Per the prior fix to eval/run_rsvqa.py: SCENE is built from
agent.vocabulary.SEGMENTATION_CLASSES (the real 19-class order), not a
simplified bucket scheme. BigEarthNet.txt's questions already reference
class names verbatim (sometimes pluralized, sometimes missing the
canonical comma, e.g. "lands principally occupied by agriculture with
significant areas of natural vegetation"), so this script extracts them
with its own class-name regex (CLASS_PATTERNS below, tolerant of
plural/singular and comma/and/or variation) rather than leaning on
agent.vocabulary's synonym map -- that map exists for free-text nouns
("roads", "grass"), not for sentences that already spell out the exact
class name. Once a class name is extracted, the normalized question this
script hands to classify_task/plan_from_query uses that exact canonical
name, so agent.planner._resolve_class's first-choice exact-match branch
resolves it immediately.

Multiple-choice questions
-----------------------------
mcq questions are answered by computing the real evidence value ONCE
(twice, for predicted/ground-truth) and then picking whichever of the 4
lettered options that value actually satisfies -- not by asking the
pipeline to "choose a letter" (it has no such tool). Presence and
adjacency mcq options are 4 different classes/pairs, each requiring its
own presence()/adjacency() call; count and area mcq options are 4
numeric/range choices compared against one count()/size() call. When zero
or more than one option is satisfied (should be rare -- the benchmark's
distractors are designed not to overlap, but our own count/size numbers
don't always land cleanly inside a distractor's boundary), the tie is
broken by picking the first (lowest-lettered) matching option if any
matched, else the row is scored unsupported rather than guessing.

What counts as "unsupported"
--------------------------------
- The class-name regex or MCQ option regex fails to parse the question
  (should be rare -- coverage was verified at ~100% on this sample before
  writing this script, but a held-out row could still surprise it).
- A count question's class resolves to "Urban fabric" or "Industrial or
  commercial units" -- the capability guardrail (agent/guardrail.py)
  always degrades these to a size() call at 10 m GSD, so no raw count is
  available to compare against a count threshold. This is a real,
  disclosed capability limit, the same one eval/run_rsvqa.py hit.
- No mcq option is satisfied by the computed value.
Each is recorded with its own `error` string in the saved CSV, not
silently skipped.

Run with the venv's own interpreter:
    venv\\Scripts\\python.exe -m eval.run_benchmark
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent import planner, tasks
from agent.dsl import PlanValidationError, validate
from agent.executor import ExecutorError, run as run_plan
from agent.planner import PlannerError, SceneDescriptor, plan_from_query
from agent.tasks import classify_task
from agent.vocabulary import SEGMENTATION_CLASSES
from evidence.schema import TraceStep
from perception import infer

# Force offline mode -- see scripts/demo_groundtruth.py's docstring for why.
planner.GOOGLE_API_KEY = None
tasks.GOOGLE_API_KEY = None

ROOT_DIR = Path(__file__).resolve().parent.parent
PATCHES_DIR = ROOT_DIR / "data" / "bench_patches"
QUESTIONS_CSV = PATCHES_DIR / "questions.csv"
OUTPUT_CSV = ROOT_DIR / "eval" / "bigearthnet_benchmark_results.csv"
GSD_METRES = 10.0  # reBEN's own documented Sentinel-2 resolution
TARGET_CATEGORIES = ("presence", "count", "area", "adjacency")

SCENE = SceneDescriptor(
    layers=["class_raster"],
    classes=dict(enumerate(SEGMENTATION_CLASSES)),
    sensor="sentinel-2",
    gsd_metres=GSD_METRES,
    bbox=(0.0, 0.0, 0.0, 0.0),  # these patches carry no geotransform
)
METADATA = {"gsd_metres": GSD_METRES}


# --- class-name extraction (tolerant of plural/comma/and-or variation) -----


def _build_class_pattern(name: str) -> re.Pattern:
    chunks = [c.strip() for c in re.split(r",\s*|\s+and\s+|\s+or\s+", name) if c.strip()]

    def word_pattern(word: str) -> str:
        base = word[:-1] if word.lower().endswith("s") and len(word) > 3 else word
        return re.escape(base) + "s?"

    chunk_patterns = [r"\s+".join(word_pattern(w) for w in chunk.split(" ")) for chunk in chunks]
    separator = r"\s*,?\s*(?:and\s+|or\s+)?"
    return re.compile(separator.join(chunk_patterns), re.IGNORECASE)


_CLASS_PATTERNS = [(name, _build_class_pattern(name)) for name in SEGMENTATION_CLASSES]


def find_classes(text: str) -> list[str]:
    """Every SEGMENTATION_CLASSES name mentioned in `text`, in reading
    order, matched at most once each occurrence (non-overlapping)."""
    matches = []
    for name, pattern in _CLASS_PATTERNS:
        m = pattern.search(text)
        if m:
            matches.append((m.start(), m.end() - m.start(), name))
    matches.sort(key=lambda item: (item[0], -item[1]))
    kept: list[str] = []
    last_end = -1
    for start, length, name in matches:
        if start >= last_end:
            kept.append(name)
            last_end = start + length
    return kept


def find_all_classes(text: str) -> list[str]:
    """Like find_classes, but every occurrence of every class (used within
    a single already-isolated mcq option, where a class name appears at
    most once anyway -- kept separate for clarity at call sites)."""
    return find_classes(text)


# --- mcq option splitting ----------------------------------------------------

_OPTION_MARKER_RE = re.compile(r"(?:^|[\s,;])([abcd])\)\s*")


def split_mcq(text: str) -> tuple[str, dict[str, str]] | tuple[None, None]:
    """"stem? a) opt1, b) opt2, c) opt3, d) opt4" -> (stem, {'a': opt1, ...}).
    Returns (None, None) if the text doesn't have exactly the 4 markers
    a)/b)/c)/d) in order."""
    markers = list(_OPTION_MARKER_RE.finditer(text))
    if [m.group(1) for m in markers] != ["a", "b", "c", "d"]:
        return None, None
    stem = text[: markers[0].start()].strip().rstrip("?").strip()
    options: dict[str, str] = {}
    for i, marker in enumerate(markers):
        start = marker.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        options[marker.group(1)] = text[start:end].strip().rstrip(",").strip()
    return stem, options


# --- numeric/threshold extraction for count and area questions --------------

_NUM_WORDS = {
    "zero": 0, "a": 1, "single": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _parse_num(token: str) -> int | None:
    token = token.strip().lower()
    if token in _NUM_WORDS:
        return _NUM_WORDS[token]
    if token.isdigit():
        return int(token)
    return None


_COUNT_BINARY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"only (?:a )?(\w+) continuous"), "eq"),
    (re.compile(r"(\w+) or less continuous"), "lte"),
    (re.compile(r"(\w+) or more (?:continuous|connected)"), "gte"),
    (re.compile(r"maximum of (\w+) continuous"), "lte"),
    (re.compile(r"fewer than (\w+)"), "lt"),
    (re.compile(r"less than (\w+)"), "lt"),
    (re.compile(r"more than (?:a )?(\w+)"), "gt"),
    (re.compile(r"at least (\w+)"), "gte"),
    (re.compile(r"at most (\w+)"), "lte"),
    (re.compile(r"exactly (\w+)"), "eq"),
]
_COUNT_MULTIPLE_RE = re.compile(r"multiple continuous areas")


def parse_count_binary(text: str) -> tuple[str, int] | None:
    t = text.lower()
    if _COUNT_MULTIPLE_RE.search(t):
        return "gt", 1
    for pattern, comparator in _COUNT_BINARY_PATTERNS:
        m = pattern.search(t)
        if m:
            n = _parse_num(m.group(1))
            if n is not None:
                return comparator, n
    return None


def parse_count_option(text: str) -> tuple[str, int] | None:
    t = text.strip().lower()
    m = re.match(r"more than (\w+)", t)
    if m:
        n = _parse_num(m.group(1))
        if n is not None:
            return "gt", n
    n = _parse_num(t)
    if n is not None:
        return "eq", n
    return None


_UNIT = r"(%|m\^2|m2|sqm|square meters)"
_AREA_WHOLE_RE = re.compile(
    r"(whole|entire) image.*(occupied|filled|covered)"
    r"|(occupied|filled|covered).*(whole|entire) image"
    r"|extend over the entire image"
    r"|cover(?:s)? the entire image"
    r"|image (?:is )?(?:completely|fully) (?:filled|covered)"
    r"|is the class .+ the only one in the image"
)
_AREA_LESS_THAN_WHOLE_RE = re.compile(r"cover(?:s)? less than the whole image")
_AREA_NOT_FULL_RE = re.compile(
    r"regions .* that are not|anything except|classes other than|some part of the image not covered by"
)
_AREA_BETWEEN_RE = re.compile(rf"between\s+([\d.]+)\s*{_UNIT}?\s+and\s+([\d.]+)\s*{_UNIT}")
_AREA_COMPOUND_RE = re.compile(rf"greater than\s+([\d.]+)\s*{_UNIT}?\s+but smaller than\s+([\d.]+)\s*{_UNIT}")
_AREA_BINARY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(rf"greater than or equal to\s+([\d.]+)\s*{_UNIT}"), "gte"),
    (re.compile(rf"equal to or above\s+([\d.]+)\s*{_UNIT}"), "gte"),
    (re.compile(rf"no less than\s+([\d.]+)\s*{_UNIT}"), "gte"),
    (re.compile(rf"at least\s+([\d.]+)\s*{_UNIT}"), "gte"),
    (re.compile(rf"no more than\s+([\d.]+)\s*{_UNIT}"), "lte"),
    (re.compile(rf"a maximum of\s+([\d.]+)\s*{_UNIT}"), "lte"),
    (re.compile(rf"at most\s+([\d.]+)\s*{_UNIT}"), "lte"),
    (re.compile(rf"less than\s+([\d.]+)\s*{_UNIT}"), "lt"),
    (re.compile(rf"smaller than\s+([\d.]+)\s*{_UNIT}"), "lt"),
    (re.compile(rf"over\s+([\d.]+)\s*{_UNIT}"), "gt"),
    (re.compile(rf"greater than\s+([\d.]+)\s*{_UNIT}"), "gt"),
    (re.compile(rf"more than\s+([\d.]+)\s*{_UNIT}"), "gt"),
    (re.compile(rf"([\d.]+)\s*{_UNIT}\s*or more"), "gte"),
]


def parse_area_binary(text: str) -> tuple[str, float, str] | tuple[str, tuple[float, float], str] | None:
    t = text.lower()
    if _AREA_WHOLE_RE.search(t):
        return "eq", 100.0, "%"
    if _AREA_LESS_THAN_WHOLE_RE.search(t):
        return "lt", 100.0, "%"
    if _AREA_NOT_FULL_RE.search(t):
        return "lt", 100.0, "%"
    m = _AREA_BETWEEN_RE.search(t)
    if m:
        return "between", (float(m.group(1)), float(m.group(3))), m.group(4)
    m = _AREA_COMPOUND_RE.search(t)
    if m:
        return "between", (float(m.group(1)), float(m.group(3))), m.group(4)
    for pattern, comparator in _AREA_BINARY_PATTERNS:
        m = pattern.search(t)
        if m:
            return comparator, float(m.group(1)), m.group(2)
    return None


_AREA_OPTION_RE = re.compile(rf"([\d.]+)\s*to\s*([\d.]+)\s*{_UNIT}")


def parse_area_option(text: str) -> tuple[float, float, str] | None:
    m = _AREA_OPTION_RE.search(text.lower())
    if m:
        return float(m.group(1)), float(m.group(2)), m.group(3)
    return None


def apply_comparator(comparator: str, value: float, threshold: float) -> bool:
    if comparator == "eq":
        return value == threshold
    if comparator == "gt":
        return value > threshold
    if comparator == "gte":
        return value >= threshold
    if comparator == "lt":
        return value < threshold
    if comparator == "lte":
        return value <= threshold
    raise ValueError(f"unknown comparator {comparator!r}")


# --- running the real pipeline for one sub-question -------------------------


@dataclass
class PipelineOutcome:
    value: Any
    tool: str
    trace: list[TraceStep]


def run_pipeline(question: str, mask: np.ndarray) -> PipelineOutcome:
    """classify_task -> plan_from_query -> validate -> executor.run, exactly
    as scripts/demo_real.py and eval/run_rsvqa.py run it."""
    classification = classify_task(question)
    plan = plan_from_query(question, SCENE)
    validate(plan)
    _, exec_trace = run_plan(plan, METADATA, mask=mask, classes=SCENE.classes)
    full_trace = list(classification.trace) + list(exec_trace)
    last_step = [s for s in exec_trace if s.task.startswith("execute:")][-1]
    return PipelineOutcome(value=last_step.output, tool=last_step.tool, trace=full_trace)


def area_in_units(size_ha: float, unit: str, total_area_m2: float) -> float:
    area_m2 = size_ha * 10_000.0
    return (area_m2 / total_area_m2) * 100.0 if unit == "%" else area_m2


# --- per-category, per-type answerers ----------------------------------------


def answer_presence_binary(class_name: str, mask: np.ndarray) -> tuple[str, PipelineOutcome]:
    outcome = run_pipeline(f"Is {class_name} present?", mask)
    return ("yes" if outcome.value else "no"), outcome


def answer_presence_mcq(options: dict[str, str], mask: np.ndarray) -> tuple[str | None, dict[str, PipelineOutcome]]:
    outcomes: dict[str, PipelineOutcome] = {}
    satisfied: list[str] = []
    for letter, option_text in options.items():
        classes = find_classes(option_text)
        if not classes:
            continue
        outcome = run_pipeline(f"Is {classes[0]} present?", mask)
        outcomes[letter] = outcome
        if outcome.value:
            satisfied.append(letter)
    predicted = satisfied[0] if satisfied else None
    return predicted, outcomes


def answer_count_binary(class_name: str, comparator: str, threshold: int,
                         mask: np.ndarray) -> tuple[str | None, PipelineOutcome]:
    outcome = run_pipeline(f"How many {class_name} are there?", mask)
    if outcome.tool != "count":
        return None, outcome  # guardrail degraded to size -- no count to compare
    return ("yes" if apply_comparator(comparator, outcome.value, threshold) else "no"), outcome


def answer_count_mcq(class_name: str, options: dict[str, str],
                      mask: np.ndarray) -> tuple[str | None, PipelineOutcome]:
    outcome = run_pipeline(f"How many {class_name} are there?", mask)
    if outcome.tool != "count":
        return None, outcome
    for letter, option_text in options.items():
        parsed = parse_count_option(option_text)
        if parsed is None:
            continue
        comparator, threshold = parsed
        if apply_comparator(comparator, outcome.value, threshold):
            return letter, outcome
    return None, outcome


def answer_area_binary(class_name: str, comparator: str, threshold, unit: str,
                        mask: np.ndarray, total_area_m2: float) -> tuple[str, PipelineOutcome]:
    outcome = run_pipeline(f"How much area of {class_name}?", mask)
    value = area_in_units(outcome.value, unit, total_area_m2)
    if comparator == "between":
        lo, hi = threshold
        result = lo <= value <= hi
    else:
        result = apply_comparator(comparator, value, threshold)
    return ("yes" if result else "no"), outcome


def answer_area_mcq(class_name: str, options: dict[str, str], mask: np.ndarray,
                     total_area_m2: float) -> tuple[str | None, PipelineOutcome]:
    outcome = run_pipeline(f"How much area of {class_name}?", mask)
    for letter, option_text in options.items():
        parsed = parse_area_option(option_text)
        if parsed is None:
            continue
        lo, hi, unit = parsed
        value = area_in_units(outcome.value, unit, total_area_m2)
        if lo <= value <= hi:
            return letter, outcome
    return None, outcome


def answer_adjacency_binary(class_a: str, class_b: str, mask: np.ndarray) -> tuple[str, PipelineOutcome]:
    outcome = run_pipeline(f"Is {class_a} near {class_b}?", mask)
    return ("yes" if outcome.value else "no"), outcome


def answer_adjacency_mcq(options: dict[str, str], mask: np.ndarray) -> tuple[str | None, dict[str, PipelineOutcome]]:
    outcomes: dict[str, PipelineOutcome] = {}
    satisfied: list[str] = []
    for letter, option_text in options.items():
        classes = find_classes(option_text)
        if len(classes) != 2:
            continue
        outcome = run_pipeline(f"Is {classes[0]} near {classes[1]}?", mask)
        outcomes[letter] = outcome
        if outcome.value:
            satisfied.append(letter)
    predicted = satisfied[0] if satisfied else None
    return predicted, outcomes


# --- main scoring loop --------------------------------------------------------


def score_question(row: pd.Series, mask: np.ndarray, total_area_m2: float) -> dict:
    category, qtype, question, ground_truth = row["category"], row["type"], row["input"], str(row["output"]).strip().lower()
    record: dict[str, Any] = {"ID": row["ID"], "patch_id": row["patch_id"], "category": category,
                               "type": qtype, "question": question, "ground_truth": ground_truth}
    try:
        if category == "presence":
            if qtype == "mcq":
                stem, options = split_mcq(question)
                if options is None:
                    raise PlannerError(f"mcq option split failed: {question!r}")
                predicted, _ = answer_presence_mcq(options, mask)
            else:
                classes = find_classes(question)
                if not classes:
                    raise PlannerError(f"no class found in presence question: {question!r}")
                predicted, _ = answer_presence_binary(classes[0], mask)

        elif category == "count":
            if qtype == "mcq":
                stem, options = split_mcq(question)
                if options is None:
                    raise PlannerError(f"mcq option split failed: {question!r}")
                classes = find_classes(stem)
                if not classes:
                    raise PlannerError(f"no class found in count mcq stem: {stem!r}")
                predicted, outcome = answer_count_mcq(classes[0], options, mask)
                if outcome.tool != "count":
                    raise PlannerError(f"guardrail degraded count to {outcome.tool}; no count to bin")
            else:
                classes = find_classes(question)
                if not classes:
                    raise PlannerError(f"no class found in count question: {question!r}")
                parsed = parse_count_binary(question)
                if parsed is None:
                    raise PlannerError(f"could not parse count threshold: {question!r}")
                comparator, threshold = parsed
                predicted, outcome = answer_count_binary(classes[0], comparator, threshold, mask)
                if outcome.tool != "count":
                    raise PlannerError(f"guardrail degraded count to {outcome.tool}; no count to bin")

        elif category == "area":
            if qtype == "mcq":
                stem, options = split_mcq(question)
                if options is None:
                    raise PlannerError(f"mcq option split failed: {question!r}")
                classes = find_classes(stem)
                if not classes:
                    raise PlannerError(f"no class found in area mcq stem: {stem!r}")
                predicted, _ = answer_area_mcq(classes[0], options, mask, total_area_m2)
            else:
                classes = find_classes(question)
                if not classes:
                    raise PlannerError(f"no class found in area question: {question!r}")
                parsed = parse_area_binary(question)
                if parsed is None:
                    raise PlannerError(f"could not parse area threshold: {question!r}")
                comparator, threshold, unit = parsed
                predicted, _ = answer_area_binary(classes[0], comparator, threshold, unit, mask, total_area_m2)

        elif category == "adjacency":
            if qtype == "mcq":
                stem, options = split_mcq(question)
                if options is None:
                    raise PlannerError(f"mcq option split failed: {question!r}")
                predicted, _ = answer_adjacency_mcq(options, mask)
            else:
                classes = find_classes(question)
                if len(classes) != 2:
                    raise PlannerError(f"expected 2 classes in adjacency question, found {len(classes)}: {question!r}")
                predicted, _ = answer_adjacency_binary(classes[0], classes[1], mask)

        else:
            raise ValueError(f"category {category!r} is out of scope for this script")

        if predicted is None:
            record.update(predicted=None, unsupported=True, correct=False,
                           error="no option/threshold satisfied by the computed evidence value")
        else:
            record.update(predicted=predicted, unsupported=False, correct=(predicted == ground_truth))
    except (PlannerError, PlanValidationError, ExecutorError) as exc:
        record.update(predicted=None, unsupported=True, correct=False, error=str(exc))
    return record


def main() -> None:
    print(__doc__)

    df = pd.read_csv(QUESTIONS_CSV)
    df = df[df["category"].isin(TARGET_CATEGORIES)].reset_index(drop=True)
    print(f"Scoring {len(df)} question(s) across {df['patch_id'].nunique()} patch(es) "
          f"from {QUESTIONS_CSV.relative_to(ROOT_DIR)} "
          f"(categories: {', '.join(TARGET_CATEGORIES)})\n")

    print("Loading model and running inference on every referenced patch ...")
    session = infer.load_model()
    config = infer.load_class_config()

    predicted_masks: dict[str, np.ndarray] = {}
    ground_truth_masks: dict[str, np.ndarray] = {}
    for patch_id in df["patch_id"].unique():
        data = np.load(PATCHES_DIR / f"{patch_id}.npz", allow_pickle=True)
        stack, gt_mask = data["stack"], data["mask"]
        result = infer.segment_image(stack[:16].astype(np.float32), "fused", session=session, config=config)
        predicted_masks[patch_id] = result.mask
        ground_truth_masks[patch_id] = gt_mask
    print(f"  -> segmented {len(predicted_masks)} patch(es)\n")

    total_area_m2 = 120 * 120 * GSD_METRES**2  # every bench patch is a 120x120 px tile

    all_records: list[dict] = []
    for mask_source, mask_lookup in (("predicted", predicted_masks), ("ground_truth", ground_truth_masks)):
        for _, row in df.iterrows():
            record = score_question(row, mask_lookup[row["patch_id"]], total_area_m2)
            record["mask_source"] = mask_source
            all_records.append(record)

    results_df = pd.DataFrame(all_records)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Full per-question results saved to {OUTPUT_CSV.relative_to(ROOT_DIR)}\n")

    for mask_source, title in (("predicted", "PREDICTED mask (real model output)"),
                                ("ground_truth", "GROUND-TRUTH mask (perfect segmentation)")):
        subset = [r for r in all_records if r["mask_source"] == mask_source]
        stats: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0, "unsupported": 0})
        for record in subset:
            s = stats[record["category"]]
            s["n"] += 1
            s["correct"] += int(record["correct"])
            s["unsupported"] += int(record["unsupported"])

        print(f"\n=== Scored against the {title} ===")
        print("| Category | N | Correct | Unsupported | Accuracy |")
        print("|---|---|---|---|---|")
        total_n = total_correct = total_unsupported = 0
        for category in TARGET_CATEGORIES:
            s = stats[category]
            if s["n"] == 0:
                continue
            accuracy = s["correct"] / s["n"]
            print(f"| {category} | {s['n']} | {s['correct']} | {s['unsupported']} | {accuracy:.1%} |")
            total_n += s["n"]
            total_correct += s["correct"]
            total_unsupported += s["unsupported"]
        overall_accuracy = total_correct / total_n if total_n else float("nan")
        print(f"| **Overall** | {total_n} | {total_correct} | {total_unsupported} | {overall_accuracy:.1%} |")


if __name__ == "__main__":
    main()
