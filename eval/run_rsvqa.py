"""Scores SatQuery AI's full pipeline against data/eval/rsvqa_lr/
rsvqa_lr_test.parquet (RSVQA-LR), broken down by RSVQA-LR's four official
question types: presence, count, comparison, rural/urban.

Per row: decode the embedded PNG -> perception/infer.py (real ONNX model)
-> classify_task -> agent.planner.plan_from_query -> agent.dsl.validate ->
agent.executor.run -- the exact same pipeline scripts/demo_real.py runs,
just driven by RSVQA-LR's questions instead of the four fixed demo ones.

RGB -> 16-channel model input, without blind zero-filling
------------------------------------------------------------
RSVQA-LR ships only an already-rendered 8-bit RGB PNG per image -- there
are no raw Sentinel-2 digital numbers to recover from it. Two decisions
follow from that, both made deliberately rather than by copying the RGB
bytes in as-is:

1. Band POSITION: the model's 16 channels are the same order verified
   against models/class_config.json's mean/std and data/demo_patches/*.npz's
   real per-channel statistics (see scripts/demo_real.py's docstring):
   channel 0 = B02 (Blue), 1 = B03 (Green), 2 = B04 (Red), ascending
   wavelength. A PNG's channel order is (R, G, B) -- copying it straight
   into channels [0, 1, 2] would silently swap Red and Blue. This script
   places each PNG channel into its correct model-band slot instead.
2. Sensor id: RSVQA-LR carries no SAR signal at all, unlike
   data/demo_patches/*.npz (real dual-pol Sentinel-1). sensor_id is set to
   "optical", not "fused" -- telling the model plainly that this input has
   no radar channels, rather than leaving it to infer that from 13 zeroed
   channels the way the very first (OSCD-based) version of
   scripts/demo_real.py did.

The remaining 13 channels (7 more Sentinel-2 bands, 2 SAR, 4 auxiliary) are
zero-filled because we truly have no data for them -- that's a real gap,
not a guess, and it's the same honest gap the original OSCD-based demo
disclosed for the same reason.

The 8-bit pixel VALUES are also not real reflectance. There is no way to
recover the original digital numbers from an already-stretched, gamma-
corrected visualisation PNG. RGB_TO_DN_SCALE (3000/255) is a documented,
disclosed approximation -- a round number in the same ballpark as
class_config.json's per-band means (a few hundred to a few thousand) --
not a calibrated inverse of whatever stretch produced the PNG. Every
segmentation (and therefore every answer) run through this scaling should
be read as approximate, not scientifically precise, per CLAUDE.md's
"No pixel, no claim": the claim here is "this is our best honest guess at
the input", not "this is the true surface reflectance".

Full 19-class vocabulary -- no more 4-bucket collapse
--------------------------------------------------------
An earlier version of this script simplified the model's 19 real
BigEarthNet classes down to 4 buckets (land/water/forest/building) before
answering anything, which made 54% of the first 200 rows "unsupported":
any question about roads, grass, residential areas, meadows, parks,
heaths, or pitches got refused even though the model predicts a real
class for most of those things. This version scores directly against the
predicted mask's real 19-class output (agent.vocabulary.SEGMENTATION_
CLASSES) and resolves nouns through agent/vocabulary.py's NOUN_TO_CLASS
synonym map -- the same map agent/planner.py's keyword fallback now
consults, so this script and the planner can never disagree about what a
noun means.

That map is still many-to-one (evidence/ops.py's tools take a single
class_id -- there's no "union of classes" tool call), and it still can't
invent classes the model was never trained to predict: BigEarthNet's
19-class reduction of CORINE Land Cover has no dedicated road class and
no sport/leisure-facility class at all (the full ~44-class CORINE
nomenclature does; this model's output does not). "roads" and "parks"/
"pitches" are mapped to the closest real class ("Urban fabric") rather
than refused -- see agent/vocabulary.py's docstring for the full
reasoning and every mapping. A residual, smaller set of questions
(shape/size qualifiers like "circular", "rectangular" that no class in
this vocabulary encodes at all) remain genuinely unanswerable and are
still scored "unsupported", not silently guessed.

The answer-format adapter this task specifically asked for
--------------------------------------------------------------
- presence / comparison: a bool from evidence/ops.py is mapped to the
  fixed vocabulary "yes"/"no" RSVQA-LR's ground truth already uses.
- rural/urban: no such tool exists anywhere in agent/registry.py. This
  script adds a small, disclosed heuristic on top of a real computed
  number: "how many buildings are there?" always triggers the capability
  guardrail at 10 m GSD (buildings, 10 m, are never resolvable against the
  25 m minimum), degrading to a real evidence/ops.py size() call; the
  built-up area as a fraction of the whole scene is thresholded at
  BUILDING_FRACTION_URBAN_THRESHOLD (0.05, arbitrary and disclosed, not
  fit to this benchmark) to answer "urban" or "rural".
- count: THIS is the adapter the task is really about. RSVQA-LR's ground
  truth is an exact integer, but evidence/ops.py's count() counts
  connected components of a land-cover raster -- a fundamentally
  different quantity than an annotator's object-instance count. Scoring
  an exact predicted number against an exact ground-truth number would
  make nearly every count question wrong by construction, regardless of
  whether the qualitative answer ("a lot" vs "none") was right. This is
  not a workaround invented for this script: RSVQA-LR's OWN published
  evaluation protocol never compares exact counts either -- it buckets
  both sides into fixed bins first. bin_count() reproduces exactly that
  protocol (0 / between 1 and 10 / between 11 and 100 / between 101 and
  1000 / more than 1000) before comparing. When a count question resolves
  to "building", the capability guardrail degrades it to a size() call in
  hectares before this adapter ever sees it -- binning a hectare figure
  into these count buckets would silently compare two different physical
  quantities, so those rows are scored unsupported instead, not binned.
- comparison: no "compare two counts" tool exists in agent/registry.py
  either. This script decomposes "are there more/less/the same number of
  X than Y" into TWO independent "how many {X} are there?" /
  "how many {Y} are there?" queries, each run through the exact same real
  classify_task -> planner -> validate -> executor pipeline, and compares
  the two already-computed evidence numbers with a plain >, <, or ==. No
  comparison logic is invented beyond that. One caveat this script does
  NOT paper over: if one side is "building" and the other isn't, the
  capability guardrail degrades only the building side from a count to a
  size in hectares (see above) -- so a small number of comparison rows
  compare a component count against a hectare figure, not like for like.
  Those rows are flagged `mixed_units=True` in the saved CSV rather than
  silently treated as equivalent to the rest.

Run with the venv's own interpreter:
    venv\\Scripts\\python.exe -m eval.run_rsvqa
"""

import hashlib
import io
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

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
PARQUET_PATH = ROOT_DIR / "data" / "eval" / "rsvqa_lr" / "rsvqa_lr_test.parquet"
OUTPUT_CSV = ROOT_DIR / "eval" / "rsvqa_lr_results_19class.csv"
NUM_ROWS = 200
GSD_METRES = 10.0  # given: RSVQA-LR images are ~10 m GSD

# The real 19-class segmentation vocabulary, id:name -- NOT the old
# 4-bucket land/water/forest/building scheme. See the module docstring.
SCENE = SceneDescriptor(
    layers=["class_raster"],
    classes=dict(enumerate(SEGMENTATION_CLASSES)),
    sensor="sentinel-2",
    gsd_metres=GSD_METRES,
    bbox=(0.0, 0.0, 0.0, 0.0),  # RSVQA-LR carries no geotransform
)
METADATA = {"gsd_metres": GSD_METRES}

# See the module docstring's "RGB -> 16-channel model input" section.
RGB_TO_DN_SCALE = 3000.0 / 255.0
BUILDING_FRACTION_URBAN_THRESHOLD = 0.05


def build_model_input(rgb_hwc: np.ndarray) -> np.ndarray:
    """(256, 256, 3) uint8 RGB -> (16, 256, 256) float32 model input.
    Channels 0/1/2 (B02 Blue/B03 Green/B04 Red) get the PNG's B/G/R
    channels respectively, rescaled by RGB_TO_DN_SCALE; the other 13
    channels are zero-filled. See the module docstring."""
    reflectance = rgb_hwc.astype(np.float32) * RGB_TO_DN_SCALE
    channels = np.zeros((16, *rgb_hwc.shape[:2]), dtype=np.float32)
    channels[0] = reflectance[:, :, 2]  # B02 Blue <- PNG's B channel
    channels[1] = reflectance[:, :, 1]  # B03 Green <- PNG's G channel
    channels[2] = reflectance[:, :, 0]  # B04 Red <- PNG's R channel
    return channels


def image_hash(image_cell: dict) -> str:
    return hashlib.md5(image_cell["bytes"]).hexdigest()


def build_mask_cache(df: pd.DataFrame, session, config: infer.ClassConfig) -> dict[str, np.ndarray]:
    """One inference call per DISTINCT image -- many RSVQA-LR questions
    share the same image (in the first 200 rows, 200 questions reference
    only 11 distinct images), so this avoids re-running the model 200
    times for 11 images' worth of actual pixels."""
    cache: dict[str, np.ndarray] = {}
    for image_cell in df["image"]:
        h = image_hash(image_cell)
        if h in cache:
            continue
        rgb = np.array(Image.open(io.BytesIO(image_cell["bytes"])).convert("RGB"))
        model_input = build_model_input(rgb)
        result = infer.segment_image(model_input, "optical", session=session, config=config)
        cache[h] = result.mask  # the real 19-class raw mask -- no simplification
    return cache


# --- question-type classification (RSVQA-LR's own 4 official types) --------


def classify_question_type(question: str) -> str:
    q = question.lower()
    if "rural" in q or "urban" in q:
        return "rural_urban"
    if any(w in q for w in (" more ", " less ", "equal to the number", "same number", "compare")):
        return "comparison"
    if q.startswith("how many") or q.startswith("what is the number of") or q.startswith("what is the amount of"):
        return "count"
    if q.startswith("is there") or q.startswith("are there") or "present" in q:
        return "presence"
    return "unknown"


# --- phrasing normalization: paraphrase -> the exact form agent/planner.py's
# --- offline keyword fallback already parses (no change to that module) ----

_COUNT_REWRITE_RE = re.compile(r"^what is the (?:number|amount) of (.+?)\??$", re.IGNORECASE)


def normalize_count_question(question: str) -> str:
    m = _COUNT_REWRITE_RE.match(question.strip())
    if m:
        return f"how many {m.group(1)} are there?"
    return question


_MORE_RE = re.compile(r"^are there more (.+?) than (.+?)\??$", re.IGNORECASE)
_LESS_RE = re.compile(r"^are there less (.+?) than (.+?)\??$", re.IGNORECASE)
_EQUAL_RE = re.compile(r"^is the number of (.+?) equal to the number of (.+?)\??$", re.IGNORECASE)


def parse_comparison(question: str) -> tuple[str, str, str] | None:
    q = question.strip()
    for pattern, comparator in ((_MORE_RE, "more"), (_LESS_RE, "less"), (_EQUAL_RE, "equal")):
        m = pattern.match(q)
        if m:
            return m.group(1).strip(), m.group(2).strip(), comparator
    return None


# --- running the real pipeline for one sub-question -------------------------


@dataclass
class PipelineOutcome:
    value: Any  # the tool's raw output: int (count), float (size, ha), or bool
    tool: str
    trace: list[TraceStep]


def run_pipeline(question: str, mask: np.ndarray) -> PipelineOutcome:
    """classify_task -> plan_from_query -> validate -> executor.run, exactly
    as scripts/demo_real.py runs it. Raises PlannerError / PlanValidationError
    / ExecutorError if the query can't be resolved -- callers decide how to
    score that (this script always scores it as unsupported/wrong)."""
    classification = classify_task(question)
    plan = plan_from_query(question, SCENE)
    validate(plan)
    _, exec_trace = run_plan(plan, METADATA, mask=mask, classes=SCENE.classes)
    full_trace = list(classification.trace) + list(exec_trace)
    last_step = [s for s in exec_trace if s.task.startswith("execute:")][-1]
    return PipelineOutcome(value=last_step.output, tool=last_step.tool, trace=full_trace)


# --- answer-format adapter ---------------------------------------------------


def bin_count(n: float) -> str:
    """RSVQA-LR's own published evaluation protocol -- see the module
    docstring's "answer-format adapter" section."""
    n = round(n)
    if n <= 0:
        return "0"
    if n <= 10:
        return "between 1 and 10"
    if n <= 100:
        return "between 11 and 100"
    if n <= 1000:
        return "between 101 and 1000"
    return "more than 1000"


def normalize_bool(value: bool) -> str:
    return "yes" if value else "no"


# --- per-type handlers --------------------------------------------------------


def answer_count(question: str, mask: np.ndarray) -> PipelineOutcome:
    return run_pipeline(normalize_count_question(question), mask)


def answer_presence(question: str, mask: np.ndarray) -> PipelineOutcome:
    return run_pipeline(question, mask)


def answer_comparison(question: str, mask: np.ndarray) -> tuple[bool, PipelineOutcome, PipelineOutcome]:
    parsed = parse_comparison(question)
    if parsed is None:
        raise PlannerError(f"comparison phrasing not recognised: {question!r}")
    object_a, object_b, comparator = parsed
    outcome_a = run_pipeline(f"how many {object_a} are there?", mask)
    outcome_b = run_pipeline(f"how many {object_b} are there?", mask)
    if comparator == "more":
        result = outcome_a.value > outcome_b.value
    elif comparator == "less":
        result = outcome_a.value < outcome_b.value
    else:
        result = outcome_a.value == outcome_b.value
    return result, outcome_a, outcome_b


def answer_rural_urban(mask: np.ndarray) -> tuple[str, float, PipelineOutcome]:
    outcome = run_pipeline("how many buildings are there?", mask)
    total_area_ha = mask.size * METADATA["gsd_metres"] ** 2 / 10_000.0
    building_fraction = outcome.value / total_area_ha if outcome.tool == "size" else float("nan")
    label = "urban" if building_fraction >= BUILDING_FRACTION_URBAN_THRESHOLD else "rural"
    return label, building_fraction, outcome


# --- main scoring loop --------------------------------------------------------


def score_row(question: str, qtype: str, ground_truth: str, mask: np.ndarray) -> dict:
    record: dict[str, Any] = {"question": question, "qtype": qtype, "ground_truth": ground_truth}
    try:
        if qtype == "count":
            outcome = answer_count(question, mask)
            if outcome.tool != "count":
                # The capability guardrail degraded this to a size() call
                # (e.g. the class resolved to "Urban fabric" or "Industrial
                # or commercial units", never countable at 10 m GSD) --
                # outcome.value is now hectares, not an object count.
                # Binning a hectare figure into RSVQA-LR's
                # count bins would compare two different physical
                # quantities and call it scoring; the honest answer is
                # that this pipeline cannot produce a count here at all.
                record.update(raw_value=outcome.value, tool=outcome.tool, predicted=None,
                               unsupported=True, correct=False,
                               error=f"guardrail degraded count to {outcome.tool}; "
                                     f"no object count available to bin")
                return record
            predicted_bin = bin_count(outcome.value)
            record.update(raw_value=outcome.value, tool=outcome.tool, predicted=predicted_bin)
            correct = predicted_bin == bin_count(float(ground_truth))
        elif qtype == "presence":
            outcome = answer_presence(question, mask)
            predicted = normalize_bool(bool(outcome.value))
            record.update(raw_value=outcome.value, tool=outcome.tool, predicted=predicted)
            correct = predicted == ground_truth
        elif qtype == "comparison":
            result, outcome_a, outcome_b = answer_comparison(question, mask)
            predicted = normalize_bool(result)
            record.update(
                predicted=predicted, value_a=outcome_a.value, tool_a=outcome_a.tool,
                value_b=outcome_b.value, tool_b=outcome_b.tool,
                mixed_units=outcome_a.tool != outcome_b.tool,
            )
            correct = predicted == ground_truth
        elif qtype == "rural_urban":
            predicted, building_fraction, outcome = answer_rural_urban(mask)
            record.update(predicted=predicted, building_fraction=building_fraction, tool=outcome.tool)
            correct = predicted == ground_truth
        else:
            record.update(predicted=None)
            correct = False
        record["unsupported"] = False
    except (PlannerError, PlanValidationError, ExecutorError) as exc:
        record.update(predicted=None, unsupported=True, error=str(exc))
        correct = False
    record["correct"] = correct
    return record


def main() -> None:
    print(__doc__)

    df = pd.read_parquet(PARQUET_PATH).iloc[:NUM_ROWS].reset_index(drop=True)
    df["qtype"] = df["question"].apply(classify_question_type)

    n_distinct_images = df["image"].apply(lambda cell: image_hash(cell)).nunique()
    print(f"Scoring {len(df)} question(s) over {n_distinct_images} distinct image(s) "
          f"from {PARQUET_PATH.relative_to(ROOT_DIR)}\n")

    print("Loading model and building the predicted-mask cache ...")
    session = infer.load_model()
    config = infer.load_class_config()
    mask_cache = build_mask_cache(df, session, config)
    print(f"  -> segmented {len(mask_cache)} distinct image(s)\n")

    rows_out = []
    for _, row in df.iterrows():
        mask = mask_cache[image_hash(row["image"])]
        ground_truth = str(row["answer"]).strip().lower()
        rows_out.append(score_row(row["question"], row["qtype"], ground_truth, mask))

    results_df = pd.DataFrame(rows_out)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Full per-question results (plans, raw values, errors) saved to "
          f"{OUTPUT_CSV.relative_to(ROOT_DIR)}\n")

    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0, "unsupported": 0})
    for record in rows_out:
        s = stats[record["qtype"]]
        s["n"] += 1
        s["correct"] += int(record["correct"])
        s["unsupported"] += int(record["unsupported"])

    print("| Question type | N | Correct | Unsupported | Accuracy |")
    print("|---|---|---|---|---|")
    total_n = total_correct = total_unsupported = 0
    for qtype in ("presence", "count", "comparison", "rural_urban"):
        s = stats[qtype]
        if s["n"] == 0:
            continue
        accuracy = s["correct"] / s["n"]
        print(f"| {qtype} | {s['n']} | {s['correct']} | {s['unsupported']} | {accuracy:.1%} |")
        total_n += s["n"]
        total_correct += s["correct"]
        total_unsupported += s["unsupported"]
    overall_accuracy = total_correct / total_n if total_n else float("nan")
    print(f"| **Overall** | {total_n} | {total_correct} | {total_unsupported} | {overall_accuracy:.1%} |")


if __name__ == "__main__":
    main()
