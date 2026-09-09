"""Turns a natural-language query into a validated agent.dsl.Plan.

Per CLAUDE.md: the LLM only classifies the task and emits a plan of tool
calls -- it never sees the image and never computes the answer itself. This
module sends Gemini the query plus a text-only scene descriptor (available
layers, class list, sensor, GSD, bbox) and gets back a Plan as JSON. The
image itself is never sent.

Three layers of resilience, in order:
  1. Disk cache (question+scene -> plan): checked before any API call, so a
     rehearsed query returns instantly and works fully offline.
  2. Gemini, with up to `max_retries` attempts: an invalid response (bad
     JSON, or a plan that fails agent.dsl.validate()) is fed back to the
     model along with the validation error and re-asked.
  3. A deterministic keyword fallback parser, used when the API itself is
     unavailable (no API key configured, or the call raises) -- covers
     "how many X", "how much area of X", "is X near/adjacent to/within N
     metres of Y", and "is X present" / "is there X".

    from agent.planner import SceneDescriptor, plan_from_query

    scene = SceneDescriptor(
        layers=["sentinel2_10m_rgb", "class_raster"],
        classes={0: "land", 1: "water", 2: "forest", 3: "urban", 4: "agricultural"},
        sensor="sentinel-2", gsd_metres=10.0, bbox=(77.5, 12.9, 77.7, 13.1),
    )
    plan = plan_from_query("How many water bodies are there?", scene)
"""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field

from agent.dsl import Plan, validate
from agent.registry import REGISTRY
from agent.tasks import _CHANGE_WORDS, _CROSS_MODAL_WORDS, _DESCRIBE_WORDS
from agent.vocabulary import resolve_noun

AGENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = AGENT_DIR.parent

load_dotenv(ROOT_DIR / ".env")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or None

DEFAULT_CACHE_DIR = AGENT_DIR / ".cache" / "plans"
GEMINI_MODEL = "gemini-2.5-flash"
MAX_RETRIES = 3
DEFAULT_MIN_AREA_M2 = 0.0


class PlannerError(RuntimeError):
    """Raised when no valid Plan could be produced: Gemini exhausted its
    retries, or the keyword fallback couldn't parse the query."""


class SceneDescriptor(BaseModel):
    """Everything the planner is told about the image -- never the pixels
    themselves. `classes` maps the class-ID raster's integer ids to their
    human-readable names, so the model (or the keyword fallback) can turn
    "water" into the class_id a tool call actually needs."""

    model_config = ConfigDict(extra="forbid")

    layers: list[str] = Field(..., description="Available data layers, e.g. ['sentinel2_10m_rgb', 'class_raster'].")
    classes: dict[int, str] = Field(..., description="class_id -> class name for this scene's raster.")
    sensor: str = Field(..., description="e.g. 'sentinel-2', 'sentinel-1'.")
    gsd_metres: float = Field(..., description="Ground sample distance, in metres.")
    bbox: tuple[float, float, float, float] = Field(
        ..., description="(min_lon, min_lat, max_lon, max_lat)."
    )


# --- Few-shot examples: 15 questions in BigEarthNet.txt style --------------
# Illustrative class scheme for the examples ONLY. The real scene's own
# `classes` mapping (passed at call time) is what the model must actually
# use -- the prompt says so explicitly, right next to these.

_EXAMPLE_CLASSES = {0: "land", 1: "water", 2: "forest", 3: "urban", 4: "agricultural"}

FEW_SHOT_EXAMPLES: list[dict] = [
    # presence
    {"question": "Is there any water present in this patch?",
     "plan": [{"id": "s1", "tool": "presence", "parameters": {"class_id": 1, "min_area_m2": 0}}]},
    {"question": "Does this scene contain forest cover?",
     "plan": [{"id": "s1", "tool": "presence", "parameters": {"class_id": 2, "min_area_m2": 0}}]},
    {"question": "Is agricultural land present in this image?",
     "plan": [{"id": "s1", "tool": "presence", "parameters": {"class_id": 4, "min_area_m2": 0}}]},
    {"question": "Is there a significant urban area in this patch?",
     "plan": [{"id": "s1", "tool": "presence", "parameters": {"class_id": 3, "min_area_m2": 10000}}]},
    # count
    {"question": "How many distinct water bodies are visible in this image?",
     "plan": [{"id": "s1", "tool": "count", "parameters": {"class_id": 1, "min_area_m2": 500}}]},
    {"question": "How many separate forest patches are there?",
     "plan": [{"id": "s1", "tool": "count", "parameters": {"class_id": 2, "min_area_m2": 1000}}]},
    {"question": "How many urban clusters larger than one hectare appear in the scene?",
     "plan": [{"id": "s1", "tool": "count", "parameters": {"class_id": 3, "min_area_m2": 10000}}]},
    {"question": "How many agricultural fields can be counted in this patch?",
     "plan": [{"id": "s1", "tool": "count", "parameters": {"class_id": 4, "min_area_m2": 2000}}]},
    # size
    {"question": "What is the total area covered by water in this image?",
     "plan": [{"id": "s1", "tool": "size", "parameters": {"class_id": 1}}]},
    {"question": "How much of this patch is agricultural land, in hectares?",
     "plan": [{"id": "s1", "tool": "size", "parameters": {"class_id": 4}}]},
    {"question": "What is the total forested area in this scene?",
     "plan": [{"id": "s1", "tool": "size", "parameters": {"class_id": 2}}]},
    {"question": "How much area does urban development occupy in this patch?",
     "plan": [{"id": "s1", "tool": "size", "parameters": {"class_id": 3}}]},
    # adjacency
    {"question": "Is the water body directly adjacent to the urban area?",
     "plan": [{"id": "s1", "tool": "adjacency", "parameters": {"class_a": 1, "class_b": 3, "distance_m": 0}}]},
    {"question": "Is agricultural land within 50 metres of the forest?",
     "plan": [{"id": "s1", "tool": "adjacency", "parameters": {"class_a": 4, "class_b": 2, "distance_m": 50}}]},
    {"question": "Is the urban area within 100 metres of a water body?",
     "plan": [{"id": "s1", "tool": "adjacency", "parameters": {"class_a": 3, "class_b": 1, "distance_m": 100}}]},
]

assert len(FEW_SHOT_EXAMPLES) == 15


# --- Prompt construction -----------------------------------------------------


def _type_name(ptype: Any) -> str:
    """`ptype.__name__` for a real type (int, str, ...); repr() for
    agent.registry.CLASS_ID, the int-or-list[int] sentinel, which isn't a
    type at all."""
    return getattr(ptype, "__name__", None) or repr(ptype)


def _tool_schema_text() -> str:
    """Tool descriptions generated from the live registry, so the prompt
    can never drift out of sync with agent/registry.py."""
    lines = []
    for tool in REGISTRY.values():
        params = ", ".join(f"{name}: {_type_name(ptype)}" for name, ptype in tool.permitted_parameters.items())
        lines.append(f"- {tool.name}({params or 'no parameters'}) -- {tool.description}")
    return "\n".join(lines)


def _few_shot_text() -> str:
    blocks = [f"Q: {ex['question']}\nA: {json.dumps(ex['plan'])}" for ex in FEW_SHOT_EXAMPLES]
    return (
        f"These examples use an illustrative class scheme ({_EXAMPLE_CLASSES}) -- "
        "for the examples ONLY. Use the real scene's classes given below for your "
        "actual answer.\n\n" + "\n\n".join(blocks)
    )


def _build_prompt(
    query: str,
    scene: SceneDescriptor,
    *,
    prior_plan_text: str | None = None,
    prior_error: str | None = None,
) -> str:
    parts = [
        "You are the planner for SatQuery AI, a satellite-imagery question answering system.",
        "You never compute the answer yourself. Your only job is to emit a Plan: a JSON array "
        'of tool-call steps, each shaped {"id": <string>, "tool": <tool name>, "parameters": {...}}.',
        "Only use the tools listed below, and only their listed parameters -- anything else is rejected.",
        "",
        "Available tools:",
        _tool_schema_text(),
        "",
        "Scene (use THESE class ids, not the examples'):",
        f"  sensor: {scene.sensor}",
        f"  gsd_metres: {scene.gsd_metres}",
        f"  bbox: {scene.bbox}",
        f"  layers: {scene.layers}",
        f"  classes (id: name): {scene.classes}",
        "",
        _few_shot_text(),
        "",
        f"Question: {query}",
    ]
    if prior_plan_text is not None:
        parts += [
            "",
            "Your previous answer was invalid. Fix it and answer again.",
            f"Previous answer: {prior_plan_text}",
            f"Validation error: {prior_error}",
        ]
    parts.append("")
    parts.append("Respond with ONLY the JSON array of steps -- no markdown, no explanation.")
    return "\n".join(parts)


def _extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = re.sub(r"^json\s*\n", "", text, flags=re.IGNORECASE)
    return json.loads(text)


def _call_gemini(client: Any, model: str, prompt: str) -> str:
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0),
    )
    return response.text


# --- Disk cache: question+scene -> plan --------------------------------------


def _cache_key(query: str, scene: SceneDescriptor) -> str:
    payload = query.strip().lower() + "|" + scene.model_dump_json()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_cache(path: Path) -> Plan | None:
    if not path.exists():
        return None
    try:
        plan = Plan.model_validate_json(path.read_text(encoding="utf-8"))
        validate(plan)
        return plan
    except Exception:
        return None  # corrupt/stale cache entry: treat as a miss, not a crash


def _write_cache(path: Path, plan: Plan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan.model_dump_json(), encoding="utf-8")


# --- Deterministic keyword fallback (used when the API is unavailable) -----

_FILLER = r"(?:\s+(?:is there|are there|are visible|can be counted))?"
_COUNT_RE = re.compile(rf"^how many (.+?){_FILLER}$")
_SIZE_AREA_OF_RE = re.compile(
    rf"^(?:how much area (?:of|is)|what is the (?:total )?area (?:of|covered by)|"
    rf"total area (?:of|covered by)) (.+?){_FILLER}$"
)
# a looser fallback for phrasing that skips the word "area" entirely, e.g.
# "how much forest is there in hectares?"
_SIZE_HOW_MUCH_RE = re.compile(r"^how much (.+?)(?:\s+is there)?(?:,?\s*in hectares)?$")
_ADJACENCY_WITHIN_RE = re.compile(
    r"^is (?:the |a |an |any )?(.+?) within (\d+(?:\.\d+)?)\s*(?:m|metres|meters) of "
    r"(?:the |a |an |any )?(.+)$"
)
_ADJACENCY_NEAR_RE = re.compile(
    r"^is (?:the |a |an |any )?(.+?) (?:near|adjacent to) (?:the |a |an |any )?(.+)$"
)
_PRESENCE_PRESENT_RE = re.compile(r"^is (?:there )?(?:any )?(?:a |an |the )?(.+?) present$")
_PRESENCE_THERE_RE = re.compile(r"^is there (?:any )?(?:a |an |the )?(.+)$")
_CONTAIN_RE = re.compile(r"^does (?:this|the) (?:scene|image|patch) contain (?:any )?(.+)$")


def _resolve_class(phrase: str, scene: SceneDescriptor) -> int | list[int]:
    phrase = phrase.strip().lower()
    for class_id, name in scene.classes.items():
        if name.lower() == phrase:
            return class_id
    for class_id, name in scene.classes.items():
        if name.lower() == phrase.rstrip("s") or f"{name.lower()}s" == phrase:
            return class_id
    for class_id, name in scene.classes.items():
        if name.lower() in phrase:
            return class_id
    # a shared word-stem prefix, to catch paraphrases a plain substring
    # check misses (e.g. "built-up area" vs a scene's "building" class --
    # both start "buil", but neither contains the other).
    for class_id, name in scene.classes.items():
        name_l = name.lower()
        prefix_len = min(len(name_l), 4)
        if prefix_len >= 4 and name_l[:prefix_len] in phrase:
            return class_id
    # last resort: agent.vocabulary's natural-language synonym map, for a
    # noun that doesn't literally overlap with any of the scene's own
    # class names at all (e.g. "roads", "grass", "residential" against the
    # real 19-class BigEarthNet vocabulary -- see agent/vocabulary.py).
    # A generic noun ("forest", "water") resolves to a LIST of every real
    # class it covers, not just one -- see evidence/ops.py's docstring for
    # why picking one arbitrarily was a real bug, not a simplification.
    # Only useful when the scene's own classes actually include the
    # resolved name(s); a scene built from an unrelated class scheme
    # simply won't match here either, and falls through to the error below.
    canonical_name = resolve_noun(phrase)
    if canonical_name is not None:
        names = canonical_name if isinstance(canonical_name, list) else [canonical_name]
        resolved_ids = [
            class_id
            for name in names
            for class_id, existing_name in scene.classes.items()
            if existing_name.lower() == name.lower()
        ]
        if resolved_ids:
            return resolved_ids if isinstance(canonical_name, list) else resolved_ids[0]
    raise PlannerError(f"keyword fallback: could not match {phrase!r} to a class in the scene")


def _keyword_fallback(query: str, scene: SceneDescriptor) -> Plan:
    q = query.strip().rstrip("?").strip().lower()

    # The 'cross_modal' tool (tools/cross_modal.py) likewise takes no
    # required parameters -- it always segments whatever raw stack the
    # executor was given, three ways. _CROSS_MODAL_WORDS is agent.tasks's
    # own list (shared, not duplicated), checked first so this and
    # classify_task's own fallback can never disagree about what counts as
    # a cross-modal question.
    if any(word in q for word in _CROSS_MODAL_WORDS):
        return _fallback_plan("cross_modal", {})

    # The 'change' tool (evidence/change.py) takes no parameters at all --
    # it always operates on whatever before/after rasters the executor was
    # given -- so any change-phrased query maps to the same single-step
    # plan regardless of exact wording. _CHANGE_WORDS is agent.tasks's own
    # list (shared, not duplicated) so this and classify_task's keyword
    # fallback can never disagree about what counts as a change question.
    if any(word in q for word in _CHANGE_WORDS):
        return _fallback_plan("change", {})

    # The 'caption' tool likewise takes no parameters -- its whole job is
    # describing whatever the mask actually contains, per
    # tools/caption.py. _DESCRIBE_WORDS is agent.tasks's own list, checked
    # after change words for the same reason classify_task's own fallback
    # does: "describe what changed" is a change question, not a caption one.
    if any(word in q for word in _DESCRIBE_WORDS):
        return _fallback_plan("caption", {})

    m = _ADJACENCY_WITHIN_RE.match(q)
    if m:
        parameters = {
            "class_a": _resolve_class(m.group(1), scene),
            "class_b": _resolve_class(m.group(3), scene),
            "distance_m": float(m.group(2)),
        }
        return _fallback_plan("adjacency", parameters)

    m = _ADJACENCY_NEAR_RE.match(q)
    if m:
        parameters = {
            "class_a": _resolve_class(m.group(1), scene),
            "class_b": _resolve_class(m.group(2), scene),
            "distance_m": 0.0,
        }
        return _fallback_plan("adjacency", parameters)

    m = _COUNT_RE.match(q)
    if m:
        parameters = {"class_id": _resolve_class(m.group(1), scene), "min_area_m2": DEFAULT_MIN_AREA_M2}
        return _fallback_plan("count", parameters)

    for pattern in (_SIZE_AREA_OF_RE, _SIZE_HOW_MUCH_RE):
        m = pattern.match(q)
        if m:
            parameters = {"class_id": _resolve_class(m.group(1), scene)}
            return _fallback_plan("size", parameters)

    for pattern in (_PRESENCE_PRESENT_RE, _PRESENCE_THERE_RE, _CONTAIN_RE):
        m = pattern.match(q)
        if m:
            parameters = {"class_id": _resolve_class(m.group(1), scene), "min_area_m2": DEFAULT_MIN_AREA_M2}
            return _fallback_plan("presence", parameters)

    raise PlannerError(f"keyword fallback: could not parse query {query!r}")


def _fallback_plan(tool: str, parameters: dict) -> Plan:
    plan = Plan.model_validate([{"id": "s1", "tool": tool, "parameters": parameters}])
    validate(plan)
    return plan


# --- Main entry point ---------------------------------------------------------


def plan_from_query(
    query: str,
    scene: SceneDescriptor,
    *,
    client: Any = None,
    model: str = GEMINI_MODEL,
    max_retries: int = MAX_RETRIES,
    cache_dir: Path | None = None,
) -> Plan:
    """Turn `query` into a validated Plan for `scene`, without ever sending
    the image.

    Checks the disk cache first. If that misses and an API client is
    available (either injected via `client`, or constructible because
    GOOGLE_API_KEY is set), asks Gemini for a plan, validating each
    response against agent.dsl.validate() and feeding any error back for
    up to `max_retries` attempts. If the API call itself fails (network,
    auth, quota -- "the API is unavailable"), or no key/client is
    available at all, falls back to a deterministic keyword parser instead
    of retrying a dead connection. A successful plan, from any source, is
    written to the cache before it's returned.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_path = cache_dir / f"{_cache_key(query, scene)}.json"

    cached = _read_cache(cache_path)
    if cached is not None:
        return cached

    if client is None and GOOGLE_API_KEY is None:
        plan = _keyword_fallback(query, scene)
        _write_cache(cache_path, plan)
        return plan

    if client is None:
        client = genai.Client(api_key=GOOGLE_API_KEY)

    prior_plan_text: str | None = None
    prior_error: str | None = None
    last_error: str | None = None

    for _attempt in range(max_retries):
        prompt = _build_prompt(query, scene, prior_plan_text=prior_plan_text, prior_error=prior_error)

        try:
            text = _call_gemini(client, model, prompt)
        except Exception:
            plan = _keyword_fallback(query, scene)
            _write_cache(cache_path, plan)
            return plan

        try:
            plan = Plan.model_validate(_extract_json(text))
            validate(plan)
        except Exception as exc:
            last_error = str(exc)
            prior_plan_text = text
            prior_error = last_error
            continue

        _write_cache(cache_path, plan)
        return plan

    raise PlannerError(
        f"Gemini did not produce a valid plan for {query!r} after {max_retries} attempts: {last_error}"
    )
