"""Scores a generative VLM baseline -- Gemini looking at a true-color image
and answering in free text, no mask, no evidence/ops.py, no computation at
all -- against the SAME BigEarthNet.txt questions eval/run_benchmark.py
scores our own predicted-mask pipeline on, so the two accuracy tables sit
side by side on IDENTICAL questions rather than two different samples.

This is deliberately the opposite architecture from the rest of this
codebase: per CLAUDE.md's "No pixel, no claim", every other tool in
SatQuery AI computes its answer from evidence/ops.py over a real class-ID
mask, and the LLM only classifies/plans. Here the LLM (Gemini) IS the
answer -- exactly the baseline this project exists to beat, and exactly
why its accuracy is worth reporting honestly rather than ignored.

RGB conversion
------------------
Reuses api.scenes.render_preview_png(stack) unchanged -- the same
per-channel percentile-stretched true-color preview (stack's real
B04/B03/B02 bands, i.e. indices [2, 1, 0]) already used to render the map
preview elsewhere in this codebase. One conversion function, one source of
truth for "what does this patch look like as a picture" -- not
reimplemented here.

Prompting Gemini for a "direct answer only"
------------------------------------------------
The image (as PNG bytes) and the RAW question text -- including its
lettered a)/b)/c)/d) options for mcq rows, unmodified -- are sent together,
with a short format instruction appended (answer "yes"/"no" for a binary
question, a single letter for an mcq one) so the response can be parsed
deterministically. Gemini sees nothing this codebase's own pipeline doesn't
already show a human: the picture and the question, nothing more.

Matching logic: reused, not reinvented
-------------------------------------------
Ground truth in questions.csv is always "yes"/"no" (binary) or a single
letter a-d (mcq) -- see eval/run_benchmark.py's own docstring. This script
parses Gemini's free-text response down to exactly that same vocabulary
(parse_vlm_answer below) and then compares it to ground truth with the
identical `predicted == ground_truth` string-equality eval/run_benchmark.py
uses in score_question() -- no separate scoring rule is invented for the
VLM side. A response that doesn't contain a recognizable yes/no or letter
is scored unsupported, the same category run_benchmark.py already reports
(there: a class name or mcq option the deterministic parser couldn't
extract; here: an answer Gemini's free text didn't clearly commit to).

Only the same 4 categories are in scope
--------------------------------------------
presence, count, area (evidence/ops.py's "size" tool), and adjacency --
the same TARGET_CATEGORIES eval/run_benchmark.py scores, imported from it
directly rather than redeclared, so the two scripts can never drift out of
sync about what's in scope.

Same 100 questions, both systems
-------------------------------------
A single seeded sample of 100 questions (RANDOM_STATE, N_QUESTIONS below)
is drawn from that in-scope pool. eval/run_benchmark.py's own
score_question() is imported and re-run, unmodified, against THIS sample's
predicted masks (real model inference, run once per referenced patch, same
as run_benchmark.py) -- not against the full 654-question sample already
saved in eval/bigearthnet_benchmark_results.csv, so the two accuracy
numbers in the final table are computed over the exact same rows and are
directly comparable, not two different-sized samples that happen to be
printed side by side.

Requires a live Gemini call (GOOGLE_API_KEY) -- unlike the rest of this
codebase, there is no offline keyword-fallback substitute for "what does a
generative VLM see when it looks at this picture": that's the very thing
being measured.

Run with the venv's own interpreter:
    venv\\Scripts\\python.exe -m eval.run_vlm_baseline
"""

import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

from api.scenes import render_preview_png
from eval.run_benchmark import GSD_METRES, PATCHES_DIR, QUESTIONS_CSV, TARGET_CATEGORIES, score_question
from perception import infer

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or None
# NOT the "gemini-2.5-flash" every other Gemini-calling module in this
# codebase uses (agent/planner.py, agent/tasks.py, tools/caption.py): that
# model id now 404s for this API key ("no longer available to new users").
# The API's own error names "gemini-3.6-flash" as the replacement, but
# that model's free tier is capped at a mere 20 requests/DAY (confirmed by
# a live 429 RESOURCE_EXHAUSTED response during this script's
# development, quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier,
# quotaValue 20) -- nowhere near enough for a 100-question run. Each named
# model alias draws from its own separate quota bucket; "-lite" variants
# carry a substantially larger free-tier daily allowance, confirmed by
# testing this exact call against several candidates.
GEMINI_MODEL = "gemini-flash-lite-latest"
OUTPUT_CSV = ROOT_DIR / "eval" / "vlm_baseline_results.csv"
N_QUESTIONS = 100
RANDOM_STATE = 42  # fixed, so the sample (and therefore the table) is reproducible
REQUEST_DELAY_SECONDS = 2.0  # paced to stay under the free tier's per-minute rate limit


# --- prompting Gemini for a direct, machine-parseable answer -----------------


def _direct_answer_prompt(question: str, qtype: str) -> str:
    instruction = (
        "Look at the satellite image and answer the question. "
        "Respond with EXACTLY one word: yes or no. No explanation, no punctuation."
        if qtype == "binary" else
        "Look at the satellite image and answer the question. "
        "Respond with EXACTLY one letter: a, b, c, or d. No explanation, no punctuation."
    )
    return f"{question}\n\n{instruction}"


# A single flaky call (rate limit, transient 5xx) must not stall a
# 100-question batch for minutes -- the SDK's own default retry policy has
# no overall deadline, and was observed hanging past 2 minutes on a single
# request during this script's development. Bounded here instead: a short
# per-request timeout, few retries, so a genuinely bad call fails fast and
# is scored unsupported rather than hanging the whole run.
_HTTP_OPTIONS = types.HttpOptions(
    timeout=30_000,  # ms
    retry_options=types.HttpRetryOptions(attempts=2, initial_delay=1.0, max_delay=10.0),
)


def ask_gemini(
    client: Any, image_bytes: bytes, question: str, qtype: str, *, model: str = GEMINI_MODEL,
) -> tuple[str | None, str | None]:
    """Sends the image + question to Gemini, returns (raw_text, error) --
    exactly one is None. Never raises: an API failure is treated as "no
    answer", scored unsupported by parse_vlm_answer downstream, the same
    way eval/run_benchmark.py treats a parse failure -- but the exception
    itself is still returned (not swallowed) so a caller can tell "Gemini
    declined to commit to an answer" apart from "the call itself failed"
    (rate limit, network, quota) in the saved results."""
    prompt = _direct_answer_prompt(question, qtype)
    part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
    try:
        response = client.models.generate_content(
            model=model,
            contents=[part, prompt],
            config=types.GenerateContentConfig(temperature=0.0, http_options=_HTTP_OPTIONS),
        )
        return (response.text or "").strip(), None
    except Exception as exc:
        return None, str(exc)


# --- parsing Gemini's free text down to the ground truth's own vocabulary --

_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_LETTER_RE = re.compile(r"\b([abcd])\b", re.IGNORECASE)


def parse_vlm_answer(text: str | None, qtype: str) -> str | None:
    """Reduces a free-text Gemini response to "yes"/"no" (binary) or a
    single letter (mcq) -- the same two forms questions.csv's own `output`
    column already uses -- or None if nothing recognizable is present."""
    if not text:
        return None
    pattern = _YES_NO_RE if qtype == "binary" else _LETTER_RE
    m = pattern.search(text)
    return m.group(1).lower() if m else None


# --- main scoring loop --------------------------------------------------------


def main() -> None:
    print(__doc__)

    if GOOGLE_API_KEY is None:
        raise SystemExit(
            "GOOGLE_API_KEY is not set -- the VLM baseline needs a live Gemini "
            "call (there is no offline substitute for 'what does a generative "
            "VLM answer', unlike every other tool in this codebase)."
        )
    client = genai.Client(api_key=GOOGLE_API_KEY)

    df = pd.read_csv(QUESTIONS_CSV)
    df = df[df["category"].isin(TARGET_CATEGORIES)].reset_index(drop=True)
    sample = df.sample(n=min(N_QUESTIONS, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
    print(
        f"Sampled {len(sample)} question(s) (seed={RANDOM_STATE}) across "
        f"{sample['patch_id'].nunique()} patch(es), from the same {len(df)}-question "
        f"in-scope pool eval/run_benchmark.py scores (categories: {', '.join(TARGET_CATEGORIES)})\n"
    )

    print("Loading model and running inference on every referenced patch "
          "(for OUR system's predicted-mask numbers, and the RGB previews Gemini sees) ...")
    session = infer.load_model()
    config = infer.load_class_config()
    predicted_masks: dict[str, np.ndarray] = {}
    rgb_previews: dict[str, bytes] = {}
    for patch_id in sample["patch_id"].unique():
        data = np.load(PATCHES_DIR / f"{patch_id}.npz", allow_pickle=True)
        stack = data["stack"]
        result = infer.segment_image(stack[:16].astype(np.float32), "fused", session=session, config=config)
        predicted_masks[patch_id] = result.mask
        rgb_previews[patch_id] = render_preview_png(stack)
    print(f"  -> segmented {len(predicted_masks)} patch(es), rendered {len(rgb_previews)} RGB preview(s)\n")

    total_area_m2 = 120 * 120 * GSD_METRES**2  # every bench patch is a 120x120 px tile

    print(f"Querying Gemini ({GEMINI_MODEL}) with image + question, one live call per question, "
          f"paced {REQUEST_DELAY_SECONDS}s apart ...")
    vlm_records: list[dict] = []
    call_errors = 0
    for i, row in sample.iterrows():
        raw_text, error = ask_gemini(client, rgb_previews[row["patch_id"]], row["input"], row["type"])
        predicted = parse_vlm_answer(raw_text, row["type"])
        ground_truth = str(row["output"]).strip().lower()
        unsupported = predicted is None
        call_errors += int(error is not None)
        vlm_records.append({
            "ID": row["ID"], "patch_id": row["patch_id"], "category": row["category"], "type": row["type"],
            "question": row["input"], "ground_truth": ground_truth, "raw_response": raw_text,
            "predicted": predicted, "unsupported": unsupported, "error": error,
            "correct": (not unsupported) and predicted == ground_truth,
        })
        if (i + 1) % 10 == 0 or (i + 1) == len(sample):
            print(f"  ... {i + 1}/{len(sample)} ({call_errors} call error(s) so far)")
        if i + 1 < len(sample):
            time.sleep(REQUEST_DELAY_SECONDS)
    print()
    if call_errors:
        print(f"WARNING: {call_errors}/{len(sample)} Gemini calls raised an exception (rate limit/quota/network -- "
              f"see the 'error' column in {OUTPUT_CSV.name}), scored unsupported, NOT counted as Gemini being wrong.\n")

    vlm_df = pd.DataFrame(vlm_records)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    vlm_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Full per-question VLM results saved to {OUTPUT_CSV.relative_to(ROOT_DIR)}\n")

    print("Scoring OUR system's predicted-mask pipeline (eval.run_benchmark.score_question, "
          "unmodified) on the exact same sample, for a fair side-by-side ...\n")
    our_records = [score_question(row, predicted_masks[row["patch_id"]], total_area_m2) for _, row in sample.iterrows()]

    def _stats(records: list[dict]) -> dict[str, dict[str, int]]:
        stats: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0, "unsupported": 0})
        for record in records:
            s = stats[record["category"]]
            s["n"] += 1
            s["correct"] += int(record["correct"])
            s["unsupported"] += int(record["unsupported"])
        return stats

    our_stats = _stats(our_records)
    vlm_stats = _stats(vlm_records)

    print(f"=== SatQuery AI (predicted mask) vs. Gemini VLM baseline -- same {len(sample)}-question sample ===")
    print("| Category | N | SatQuery Accuracy | SatQuery Unsupported | Gemini VLM Accuracy | Gemini VLM Unsupported |")
    print("|---|---|---|---|---|---|")
    total_n = our_total_correct = vlm_total_correct = our_total_unsupported = vlm_total_unsupported = 0
    for category in TARGET_CATEGORIES:
        os_, vs_ = our_stats[category], vlm_stats[category]
        n = os_["n"]
        if n == 0:
            continue
        our_acc, vlm_acc = os_["correct"] / n, vs_["correct"] / n
        print(f"| {category} | {n} | {our_acc:.1%} | {os_['unsupported']} | {vlm_acc:.1%} | {vs_['unsupported']} |")
        total_n += n
        our_total_correct += os_["correct"]
        vlm_total_correct += vs_["correct"]
        our_total_unsupported += os_["unsupported"]
        vlm_total_unsupported += vs_["unsupported"]
    our_overall = our_total_correct / total_n if total_n else float("nan")
    vlm_overall = vlm_total_correct / total_n if total_n else float("nan")
    print(
        f"| **Overall** | {total_n} | {our_overall:.1%} | {our_total_unsupported} | "
        f"{vlm_overall:.1%} | {vlm_total_unsupported} |"
    )


if __name__ == "__main__":
    main()
