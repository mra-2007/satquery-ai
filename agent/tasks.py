"""Classifies a natural-language query into one of SatQuery AI's six task
types, using a short few-shot Gemini prompt.

Per CLAUDE.md: the LLM only classifies the task -- it doesn't compute
anything. This is that classification step, kept separate from planning
(agent/planner.py, which turns a query into a tool-call Plan once the task
type is known). The resulting label is always wrapped in an
evidence.schema.TraceStep, so it shows up in the execution trace exactly
like every other decision in the system.

Same two layers of resilience as agent/planner.py:
  1. Disk cache (question -> label+confidence): checked before any API
     call, so a repeated question works fully offline. Unlike the planner's
     cache, this one is keyed by the question alone -- task classification
     doesn't depend on which scene the question is about.
  2. A deterministic keyword fallback, used when the API is unavailable (no
     key configured, the call raises, or it returns something we can't
     parse) -- covers change-related phrasing, description requests,
     "where is" locating phrases, and explicit sensor names (SAR/radar).

    from agent.tasks import classify_task

    result = classify_task("How many water bodies are there?")
    result.label        # "vqa"
    result.confidence    # e.g. 0.92
    result.trace         # [TraceStep(...)] -- log this into the run's trace
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from evidence.schema import TraceStep

AGENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = AGENT_DIR.parent

load_dotenv(ROOT_DIR / ".env")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or None

DEFAULT_CACHE_DIR = AGENT_DIR / ".cache" / "tasks"
# NOT "gemini-2.5-flash" -- that model id now 404s for this API key ("no
# longer available to new users"), which meant classify_task() was
# silently hitting the except-and-fall-back-to-keyword-parser path on
# every live call, never actually reaching Gemini. gemini-flash-lite-
# latest is confirmed working (including with response_mime_type=
# "application/json", the mode this module needs) and, unlike the API
# error's own suggested replacement (gemini-3.6-flash, capped at a mere
# 20 requests/day on the free tier -- see eval/run_vlm_baseline.py's own
# docstring for how that was discovered), has enough free-tier headroom
# for real use.
GEMINI_MODEL = "gemini-flash-lite-latest"

TASK_LABELS = (
    "vqa", "caption", "grounding", "change_vqa", "change_describe", "cross_modal",
    "metadata", "conversational",
)

TASK_DESCRIPTIONS: dict[str, str] = {
    "vqa": "A factual/numeric question about a single image (counts, presence, land cover, area).",
    "caption": "A request for a free-text description of a single image.",
    "grounding": "A request to locate the image region matching a natural-language description.",
    "change_vqa": "A factual/numeric question about what changed between two dates.",
    "change_describe": "A request for a free-text description of what changed between two dates.",
    "cross_modal": "A question requiring reasoning across two sensors/modalities (e.g. optical vs SAR).",
    "metadata": "A question about the scene's own record -- location, acquisition date, sensor/"
                "platform, resolution, or which classes are present -- not the imagery itself.",
    "conversational": "A greeting or an off-topic question, not about any satellite scene.",
}


class TaskClassificationError(RuntimeError):
    """Raised when neither Gemini nor the keyword fallback could classify
    the query."""


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    trace: list[TraceStep]


# --- Few-shot examples: a short prompt, two per label -----------------------

FEW_SHOT_EXAMPLES: list[tuple[str, str]] = [
    ("How many water bodies are in this image?", "vqa"),
    ("What is the dominant land cover class here?", "vqa"),
    ("Describe this satellite image.", "caption"),
    ("What does this scene show?", "caption"),
    ("Where is the largest building in this image?", "grounding"),
    ("Point to the river crossing this patch.", "grounding"),
    ("How much forest was lost between 2019 and 2023?", "change_vqa"),
    ("Did the urban area grow over the last five years?", "change_vqa"),
    ("Describe what changed between these two images.", "change_describe"),
    ("Summarize the land-cover changes over time.", "change_describe"),
    ("Does the SAR image confirm the flooding visible in the optical image?", "cross_modal"),
    ("Compare the radar and optical imagery for this area.", "cross_modal"),
    ("Where is this scene located?", "metadata"),
    ("What sensor was used to capture this?", "metadata"),
    ("Hi there!", "conversational"),
    ("What's your favorite color?", "conversational"),
]

assert {label for _, label in FEW_SHOT_EXAMPLES} == set(TASK_LABELS)


def _build_prompt(query: str) -> str:
    labels_text = "\n".join(f"- {label}: {desc}" for label, desc in TASK_DESCRIPTIONS.items())
    examples_text = "\n".join(
        f'Q: "{question}" -> {{"label": "{label}", "confidence": 0.9}}'
        for question, label in FEW_SHOT_EXAMPLES
    )
    return (
        "Classify the following satellite-imagery query into exactly one task type.\n\n"
        f"Task types:\n{labels_text}\n\n"
        f"Examples:\n{examples_text}\n\n"
        f'Query: "{query}"\n\n'
        'Respond with ONLY a JSON object: {"label": <one of the task types above>, '
        '"confidence": <number between 0 and 1>}.'
    )


def _call_gemini(client: Any, model: str, prompt: str) -> str:
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0),
    )
    return response.text


def _parse_response(text: str) -> tuple[str, float]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = re.sub(r"^json\s*\n", "", text, flags=re.IGNORECASE)
    data = json.loads(text)
    label = data["label"]
    confidence = float(data["confidence"])
    if label not in TASK_LABELS:
        raise ValueError(f"unknown label {label!r}; expected one of {TASK_LABELS}")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence {confidence} out of range [0, 1]")
    return label, confidence


# --- Disk cache: question -> label+confidence --------------------------------


def _cache_key(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()


def _read_cache(path: Path) -> tuple[str, float] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        label, confidence = data["label"], float(data["confidence"])
        if label not in TASK_LABELS or not 0.0 <= confidence <= 1.0:
            return None
        return label, confidence
    except Exception:
        return None  # corrupt/stale cache entry: treat as a miss, not a crash


def _write_cache(path: Path, label: str, confidence: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"label": label, "confidence": confidence}), encoding="utf-8")


# --- Deterministic keyword fallback (used when the API is unavailable) -----

_CROSS_MODAL_WORDS = ("sar", "radar")
_CHANGE_WORDS = (
    "change", "changed", "difference between", "before and after", "over time",
    "between", "grew", "grow", "lost", "gained", "increase", "decrease",
)
_DESCRIBE_WORDS = ("describe", "summarize", "summarise", "what does this show", "caption")
_LOCATE_WORDS = ("where is", "point to", "locate", "show me the", "find the")

# A short, common set of greeting openers -- matched at the START of the
# (trimmed, lowercased) query, not anywhere in it, so a real question that
# happens to contain one of these words elsewhere is never misclassified.
_GREETING_RE = re.compile(
    r"^(hi|hello|hey|hiya|howdy|yo|greetings|good morning|good afternoon|good evening)\b"
)

# Metadata: a question about the SCENE'S OWN RECORD (location, acquisition
# date, sensor/platform, resolution, or which classes are present), not
# about the imagery itself. "where is this" is checked here, and BEFORE
# _LOCATE_WORDS below, specifically so it doesn't fall through to
# "grounding" -- "where is this [scene]" asks about the whole scene's own
# location, "where is the water body" asks to locate a region within it.
_METADATA_WORDS = (
    "where is this", "when was this", "acquisition date", "what date",
    "what sensor", "what satellite", "what platform",
    "what resolution", "ground sample distance", "what's the gsd", "what is the gsd",
    "what's in this scene", "what is in this scene", "what classes are", "what land cover is here",
)


def _keyword_fallback(query: str) -> tuple[str, float]:
    q = query.strip().lower()

    if _GREETING_RE.match(q):
        return "conversational", 0.9

    if any(word in q for word in _METADATA_WORDS):
        return "metadata", 0.8

    if any(word in q for word in _CROSS_MODAL_WORDS):
        return "cross_modal", 0.75

    if any(word in q for word in _CHANGE_WORDS):
        if any(word in q for word in _DESCRIBE_WORDS):
            return "change_describe", 0.75
        return "change_vqa", 0.75

    if any(word in q for word in _DESCRIBE_WORDS):
        return "caption", 0.75

    if any(word in q for word in _LOCATE_WORDS):
        return "grounding", 0.75

    # No distinctive signal at all -- still default to "vqa" here (this is
    # only a trace LABEL, not what builds the actual plan: agent/planner.py's
    # own, more detailed keyword fallback separately parses "how many X"/
    # "how much X"/"is X present" style phrasing that this coarser word list
    # never sees at all, and correctly builds a real plan for it regardless
    # of this label). Only agent/planner.py's fallback -- which actually
    # knows whether it could build a plan -- decides "conversational".
    return "vqa", 0.4


# --- Main entry point ---------------------------------------------------------


def _result(query: str, label: str, confidence: float, source: str) -> ClassificationResult:
    trace_step = TraceStep(
        task="classify_task",
        tool="agent.tasks.classify_task",
        parameters={"query": query, "source": source},
        output={"label": label},
        confidence=confidence,
    )
    return ClassificationResult(label=label, confidence=confidence, trace=[trace_step])


def classify_task(
    query: str,
    *,
    client: Any = None,
    model: str = GEMINI_MODEL,
    cache_dir: Path | None = None,
) -> ClassificationResult:
    """Classify `query` into one of TASK_LABELS.

    Checks the disk cache first (keyed by the question alone). On a miss,
    asks Gemini (using an injected `client` if given, else constructing one
    from GOOGLE_API_KEY). If the API is unavailable -- no key/client, the
    call raises, or the response can't be parsed into a valid label and
    confidence -- falls back to a deterministic keyword parser instead. A
    successful classification, from any source, is written to the cache
    and always comes back wrapped in a TraceStep (`result.trace`), so the
    label is guaranteed to appear in the execution trace.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_path = cache_dir / f"{_cache_key(query)}.json"

    cached = _read_cache(cache_path)
    if cached is not None:
        label, confidence = cached
        return _result(query, label, confidence, "cache")

    if client is None and GOOGLE_API_KEY is None:
        label, confidence = _keyword_fallback(query)
        _write_cache(cache_path, label, confidence)
        return _result(query, label, confidence, "keyword_fallback")

    if client is None:
        client = genai.Client(api_key=GOOGLE_API_KEY)

    try:
        text = _call_gemini(client, model, _build_prompt(query))
        label, confidence = _parse_response(text)
        source = "gemini"
    except Exception:
        label, confidence = _keyword_fallback(query)
        source = "keyword_fallback"

    _write_cache(cache_path, label, confidence)
    return _result(query, label, confidence, source)
