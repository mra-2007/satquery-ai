"""Deterministic captioning: every fact in the caption is computed by
evidence/ops.py over the real class-ID mask, using agent/vocabulary.py's
19-class segmentation vocabulary -- never invented by an LLM. Gemini's
only job, if it runs at all, is rewording an already-complete, already-
correct sentence into more natural prose; if it's unavailable, caption()
returns that sentence unchanged, still fully correct, just less fluent.

This mirrors how BigEarthNet.txt's own captions were generated (see
data/benchmark/bigearthnet_txt_benchmark.csv's `type == "captioning"` rows,
and the very similar 'output' strings that turn up when
data/bench_patches/questions.csv's category is unrelated but its wording
still says "The dominant feature is X, covering approximately Y square
meters ... X borders Y ..."): area totals per class, fragment counts, and
which classes are adjacent to which, stated first, described second.

Per CLAUDE.md's "No pixel, no claim": extract_facts() and
extract_adjacent_pairs() are the only two functions in this module that
touch the mask, and both call straight into evidence/ops.py -- the exact
same count()/size()/adjacency() every other tool in this codebase uses,
so there is exactly one source of truth for these numbers. render_template()
turns those facts into a sentence with no further computation. Gemini
never sees the mask, never sees the raw numbers' source, and is
explicitly instructed not to change any of them -- caption() still
returns the untouched `template_sentence` alongside whatever Gemini
produced, so a caller (or a test) can always check the two agree.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from google import genai

from agent.vocabulary import SEGMENTATION_CLASSES
from evidence import ops

TOOLS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TOOLS_DIR.parent

load_dotenv(ROOT_DIR / ".env")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or None
# NOT "gemini-2.5-flash" -- see agent/tasks.py's own GEMINI_MODEL comment:
# that model id now 404s for this API key, which meant caption()'s
# Gemini-smoothing step was silently falling straight through to
# render_template()'s deterministic sentence, unsmoothed, on every live call.
GEMINI_MODEL = "gemini-flash-lite-latest"


@dataclass
class ClassFact:
    """One present class's computed facts -- everything here came from
    evidence/ops.py, never from an LLM."""

    class_id: int
    class_name: str
    area_ha: float
    fragment_count: int


@dataclass
class CaptionResult:
    caption: str  # the final text: Gemini-smoothed if available, else template_sentence verbatim
    template_sentence: str  # the deterministic, fact-only sentence -- always correct, for audit
    facts: list[ClassFact]
    adjacent_pairs: list[tuple[str, str]]
    smoothed: bool  # whether Gemini actually produced `caption`


def extract_facts(mask: np.ndarray, metadata: dict, *, min_area_m2: float = 0.0) -> list[ClassFact]:
    """One ClassFact per SEGMENTATION_CLASSES class present in `mask`
    (area > 0), sorted by area descending -- the order a human describing
    "the dominant feature is X" would use."""
    facts = []
    for class_id, class_name in enumerate(SEGMENTATION_CLASSES):
        area_ha = ops.size(mask, class_id, metadata)
        if area_ha <= 0:
            continue
        fragment_count = ops.count(mask, class_id, min_area_m2, metadata)
        facts.append(ClassFact(class_id, class_name, area_ha, fragment_count))
    facts.sort(key=lambda fact: fact.area_ha, reverse=True)
    return facts


def extract_adjacent_pairs(mask: np.ndarray, metadata: dict, facts: list[ClassFact]) -> list[tuple[str, str]]:
    """Every unordered pair of present classes that directly touch
    (distance_m=0), via evidence/ops.py's own adjacency() -- same tool,
    same semantics, as every adjacency question elsewhere in this
    codebase."""
    pairs: list[tuple[str, str]] = []
    for i, fact_a in enumerate(facts):
        for fact_b in facts[i + 1:]:
            if ops.adjacency(mask, fact_a.class_id, fact_b.class_id, 0.0, metadata):
                pairs.append((fact_a.class_name, fact_b.class_name))
    return pairs


def render_template(facts: list[ClassFact], adjacent_pairs: list[tuple[str, str]]) -> str:
    """The deterministic, fact-only sentence -- what caption() falls back
    to verbatim whenever Gemini is unavailable."""
    if not facts:
        return "No land-cover classes were detected in this image."

    dominant = facts[0]
    sentences = [
        f"The dominant class is {dominant.class_name} ({dominant.area_ha:.2f} ha, "
        f"{dominant.fragment_count} distinct area(s))."
    ]
    for fact in facts[1:]:
        sentences.append(
            f"{fact.class_name} covers {fact.area_ha:.2f} ha ({fact.fragment_count} distinct area(s))."
        )
    if adjacent_pairs:
        pairs_text = "; ".join(f"{a} borders {b}" for a, b in adjacent_pairs)
        sentences.append(f"Adjacent classes: {pairs_text}.")
    return " ".join(sentences)


def _smooth_with_gemini(template_sentence: str, *, client: Any = None, model: str = GEMINI_MODEL) -> str | None:
    """Ask Gemini to reword `template_sentence` more fluently, changing no
    fact. Returns None (never raises) if the API is unavailable or the
    call fails -- caption() then falls back to the template unchanged."""
    if client is None and GOOGLE_API_KEY is None:
        return None
    if client is None:
        client = genai.Client(api_key=GOOGLE_API_KEY)

    prompt = (
        "Reword the following satellite-image description so it reads as one "
        "natural, fluent paragraph. Do NOT add, remove, or change any number, "
        "percentage, area, class name, or adjacency claim in it -- only improve "
        "the prose. If you cannot do this without changing a fact, return the "
        "text unchanged.\n\n"
        f"{template_sentence}"
    )
    try:
        response = client.models.generate_content(model=model, contents=prompt)
        text = (response.text or "").strip()
        return text or None
    except Exception:
        return None


def caption(
    mask: np.ndarray,
    metadata: dict,
    *,
    max_length: int | None = None,
    client: Any = None,
    model: str = GEMINI_MODEL,
    min_area_m2: float = 0.0,
) -> CaptionResult:
    """Caption `mask`: extract facts, render the deterministic template,
    then let Gemini smooth the wording (never the facts). `max_length`,
    if given, truncates the final caption to that many characters --
    applied after smoothing, deterministically, never by asking Gemini to
    self-limit.
    """
    facts = extract_facts(mask, metadata, min_area_m2=min_area_m2)
    adjacent_pairs = extract_adjacent_pairs(mask, metadata, facts)
    template_sentence = render_template(facts, adjacent_pairs)

    smoothed_text = _smooth_with_gemini(template_sentence, client=client, model=model)
    final_caption = smoothed_text if smoothed_text is not None else template_sentence
    if max_length is not None:
        final_caption = final_caption[:max_length]

    return CaptionResult(
        caption=final_caption,
        template_sentence=template_sentence,
        facts=facts,
        adjacent_pairs=adjacent_pairs,
        smoothed=smoothed_text is not None,
    )
