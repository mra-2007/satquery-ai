"""The verifier: CLAUDE.md's "No pixel, no claim", enforced on TEXT rather
than trusted on faith.

Every other tool in this codebase already computes its answer from the mask
directly (evidence/ops.py, evidence/change.py) -- there is nothing to
"verify" about a `count()` call's own return value, it IS the ground truth.
The one place a fact can genuinely drift from the mask is tools/caption.py's
`caption` field: it's Gemini's reworded version of a fact-only template
sentence, and although caption.py's prompt explicitly instructs Gemini not
to change any number, that instruction is not a guarantee. This module is
the independent, blind check that doesn't take Gemini's word for it: it
re-derives each factual claim in a piece of text straight from the mask via
evidence/ops.py, the same way the text's numbers were (should have been)
produced in the first place, and reports exactly which claims hold up.

Claim extraction is deterministic regex over a fixed, small set of
sentence shapes -- the same shapes this codebase's own tools actually
produce (tools/caption.py's render_template(), and api/rendering.py's
build_answer() terse forms like "31.77 ha of forest" or "3 water bodies")
plus a few common plain-English equivalents ("There are 3 water bodies.",
"Water is present.", "Urban fabric borders water."). It is not a general
NLP claim extractor: a sentence phrased some other way, or one that packs
two different claim types into a single "and"-joined clause, is not
guaranteed to be split apart -- only the first recognised clause shape in
such a sentence is checked. A sentence that isn't a recognised numeric/
presence/adjacency assertion at all (plain descriptive prose) is left
alone: it is not a "claim" this system can check, so it is neither
verified nor flagged, per CLAUDE.md -- only pixel-backed claims are ever
adjudicated.

A recognised claim that CAN be checked is re-tested with the same
evidence/ops.py function a plan step would use for that same question
(count/size/presence/adjacency), over the same mask, using
agent/vocabulary.py's noun-to-class resolution -- one shared vocabulary,
so this module can never disagree with the planner about what a noun
means. A claim whose class doesn't resolve at all, or whose recomputed
value contradicts what the text asserted, is FLAGGED (default) or DROPPED
(on_unverifiable="drop") from the returned verified_answer -- never
silently left in place unexamined.
"""

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np

from agent.vocabulary import SEGMENTATION_CLASSES, resolve_noun
from evidence import ops
from evidence.ops import ClassId

ClaimType = Literal["size", "count", "presence", "adjacency"]

DEFAULT_MIN_AREA_M2 = 0.0
# A recomputed hectare figure is judged against the claimed one with this
# tolerance (relative, floored absolute) -- wide enough to absorb the
# claimed text's own rounding (e.g. "31.77 ha" for a true 31.7742...), tight
# enough that a genuinely different number still fails.
SIZE_REL_TOL = 0.02
SIZE_ABS_TOL = 0.02

_CLASS_NAME_TO_ID_LOWER: dict[str, int] = {name.lower(): i for i, name in enumerate(SEGMENTATION_CLASSES)}

_CLASS_PHRASE = r"[A-Za-z][A-Za-z,\-\s]*?"
_NUMBER = r"[\d,]+\.?\d*"


@dataclass
class Claim:
    """One factual claim extracted from a generated answer, and the
    result of re-testing it against the mask."""

    text: str  # the exact substring asserting this claim
    claim_type: ClaimType
    tool: str  # "evidence.ops.<name>" -- the function re-run to check this claim
    class_name: str | None  # None only when the claimed class couldn't be resolved at all
    claimed_value: bool | int | float | None
    recomputed_value: bool | int | float | None
    passed: bool
    reason: str | None = None  # set whenever passed is False


@dataclass
class VerificationResult:
    original_answer: str
    verified_answer: str
    claims: list[Claim]

    @property
    def all_passed(self) -> bool:
        return all(claim.passed for claim in self.claims)


def _parse_number(text: str) -> float:
    return float(text.replace(",", ""))


def _resolve_class_phrase(phrase: str) -> ClassId | None:
    """Resolve a short noun phrase (e.g. "forest", "Urban fabric") to one
    or more of SEGMENTATION_CLASSES's real class ids -- an exact
    (case-insensitive) class-name match first, then
    agent.vocabulary.resolve_noun()'s synonym map, exactly like
    tools/grounding.py and agent/planner.py's keyword fallback. Returns
    None if nothing resolves."""
    phrase = phrase.strip().strip(".,;:").lower()
    if not phrase:
        return None

    if phrase in _CLASS_NAME_TO_ID_LOWER:
        return _CLASS_NAME_TO_ID_LOWER[phrase]

    resolved = resolve_noun(phrase)
    if resolved is None:
        return None
    names = resolved if isinstance(resolved, list) else [resolved]
    ids = [_CLASS_NAME_TO_ID_LOWER[name.lower()] for name in names]
    return ids if isinstance(resolved, list) else ids[0]


def _class_label(class_id: ClassId) -> str:
    if isinstance(class_id, list):
        return " / ".join(SEGMENTATION_CLASSES[i] for i in class_id)
    return SEGMENTATION_CLASSES[class_id]


def _unresolvable_claim(claim_type: ClaimType, tool: str, text: str, phrase: str) -> Claim:
    return Claim(
        text=text, claim_type=claim_type, tool=tool, class_name=None,
        claimed_value=None, recomputed_value=None, passed=False,
        reason=f"could not resolve {phrase!r} to a known land-cover class",
    )


# --- per-claim-type checkers: (mask, metadata) + parsed groups -> Claim -----


def _check_size(mask: np.ndarray, metadata: dict, text: str, class_phrase: str, claimed_ha: float) -> Claim:
    class_id = _resolve_class_phrase(class_phrase)
    if class_id is None:
        return _unresolvable_claim("size", "evidence.ops.size", text, class_phrase)

    recomputed = ops.size(mask, class_id, metadata)
    passed = abs(recomputed - claimed_ha) <= max(SIZE_ABS_TOL, SIZE_REL_TOL * claimed_ha)
    return Claim(
        text=text, claim_type="size", tool="evidence.ops.size", class_name=_class_label(class_id),
        claimed_value=claimed_ha, recomputed_value=recomputed, passed=passed,
        reason=None if passed else f"claimed {claimed_ha:.2f} ha, recomputed {recomputed:.2f} ha",
    )


def _check_count(mask: np.ndarray, metadata: dict, text: str, class_phrase: str, claimed_count: int) -> Claim:
    class_id = _resolve_class_phrase(class_phrase)
    if class_id is None:
        return _unresolvable_claim("count", "evidence.ops.count", text, class_phrase)

    recomputed = ops.count(mask, class_id, DEFAULT_MIN_AREA_M2, metadata)
    passed = recomputed == claimed_count
    return Claim(
        text=text, claim_type="count", tool="evidence.ops.count", class_name=_class_label(class_id),
        claimed_value=claimed_count, recomputed_value=recomputed, passed=passed,
        reason=None if passed else f"claimed {claimed_count}, recomputed {recomputed}",
    )


def _check_presence(mask: np.ndarray, metadata: dict, text: str, class_phrase: str) -> Claim:
    class_id = _resolve_class_phrase(class_phrase)
    if class_id is None:
        return _unresolvable_claim("presence", "evidence.ops.presence", text, class_phrase)

    recomputed = ops.presence(mask, class_id, DEFAULT_MIN_AREA_M2, metadata)
    return Claim(
        text=text, claim_type="presence", tool="evidence.ops.presence", class_name=_class_label(class_id),
        claimed_value=True, recomputed_value=recomputed, passed=recomputed,
        reason=None if recomputed else "not present in the mask",
    )


def _check_adjacency(mask: np.ndarray, metadata: dict, text: str, phrase_a: str, phrase_b: str) -> Claim:
    class_a = _resolve_class_phrase(phrase_a)
    class_b = _resolve_class_phrase(phrase_b)
    if class_a is None:
        return _unresolvable_claim("adjacency", "evidence.ops.adjacency", text, phrase_a)
    if class_b is None:
        return _unresolvable_claim("adjacency", "evidence.ops.adjacency", text, phrase_b)

    recomputed = ops.adjacency(mask, class_a, class_b, 0.0, metadata)
    class_name = f"{_class_label(class_a)} / {_class_label(class_b)}"
    return Claim(
        text=text, claim_type="adjacency", tool="evidence.ops.adjacency", class_name=class_name,
        claimed_value=True, recomputed_value=recomputed, passed=recomputed,
        reason=None if recomputed else "not adjacent in the mask",
    )


# --- sentence-level claim extraction ----------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Combined size+count, matching tools/caption.py's own template sentences
# verbatim -- two distinct shapes, not one, since the dominant-class
# sentence puts "(area ha, count distinct area(s))" right after the class
# name, while every other class's sentence states the area BEFORE the
# parenthesised count: "The dominant class is X (12.34 ha, 2 distinct
# area(s))." vs "X covers 12.34 ha (2 distinct area(s))."
_DOMINANT_SIZE_COUNT_RE = re.compile(
    rf"dominant (?:class|feature) is\s+(?P<cls>{_CLASS_PHRASE})\s*"
    rf"\(\s*(?P<area>{_NUMBER})\s*ha,\s*(?P<count>\d+)\s*distinct area",
    re.IGNORECASE,
)
_COVERS_SIZE_COUNT_RE = re.compile(
    rf"(?P<cls>{_CLASS_PHRASE})\s+covers\s+(?P<area>{_NUMBER})\s*ha\s*"
    rf"\(\s*(?P<count>\d+)\s*distinct area",
    re.IGNORECASE,
)
# Terse size forms: "31.77 ha of forest" (api/rendering.py's build_answer)
# and a standalone "X covers 12.34 ha" with no count clause.
_SIZE_HA_OF_RE = re.compile(rf"(?P<area>{_NUMBER})\s*ha\s+of\s+(?P<cls>{_CLASS_PHRASE})(?=[.,;]|$)", re.IGNORECASE)
_SIZE_COVERS_RE = re.compile(rf"(?P<cls>{_CLASS_PHRASE})\s+covers\s+(?P<area>{_NUMBER})\s*ha\b", re.IGNORECASE)
# Terse/plain count forms: "3 water bodies" (a whole terse answer) and
# "There are 3 water bodies." / "There is 1 water body."
_BARE_COUNT_RE = re.compile(rf"^(?P<count>\d+)\s+(?P<cls>[A-Za-z][A-Za-z\-\s]*)$", re.IGNORECASE)
_THERE_ARE_COUNT_RE = re.compile(
    rf"there\s+(?:are|is)\s+(?P<count>\d+)\s+(?P<cls>{_CLASS_PHRASE})(?=[.,;]|$)", re.IGNORECASE,
)
_PRESENT_RE = re.compile(rf"(?P<cls>{_CLASS_PHRASE})\s+(?:is|are)\s+present\b", re.IGNORECASE)
_THERE_IS_RE = re.compile(
    rf"there\s+(?:is|are)\s+(?:a|an|some)?\s*(?P<cls>{_CLASS_PHRASE})(?=[.,;]|$)", re.IGNORECASE,
)
_ADJACENCY_RE = re.compile(
    rf"(?P<a>{_CLASS_PHRASE})\s+(?:borders|is\s+(?:near|adjacent to|next to))\s+(?P<b>{_CLASS_PHRASE})(?=[.,;]|$)",
    re.IGNORECASE,
)


def _claims_in_sentence(mask: np.ndarray, metadata: dict, sentence: str) -> list[Claim]:
    claims: list[Claim] = []

    m = _DOMINANT_SIZE_COUNT_RE.search(sentence) or _COVERS_SIZE_COUNT_RE.search(sentence)
    if m:
        claims.append(_check_size(mask, metadata, m.group(0), m.group("cls"), _parse_number(m.group("area"))))
        claims.append(_check_count(mask, metadata, m.group(0), m.group("cls"), int(m.group("count"))))
        return claims

    m = _SIZE_HA_OF_RE.search(sentence)
    if m:
        claims.append(_check_size(mask, metadata, m.group(0), m.group("cls"), _parse_number(m.group("area"))))
    else:
        m = _SIZE_COVERS_RE.search(sentence)
        if m:
            claims.append(_check_size(mask, metadata, m.group(0), m.group("cls"), _parse_number(m.group("area"))))

    m = _THERE_ARE_COUNT_RE.search(sentence)
    if m:
        claims.append(_check_count(mask, metadata, m.group(0), m.group("cls"), int(m.group("count"))))
    else:
        m = _BARE_COUNT_RE.match(sentence.strip().rstrip(".!?"))
        if m:
            claims.append(_check_count(mask, metadata, m.group(0), m.group("cls"), int(m.group("count"))))

    if not claims:
        m = _PRESENT_RE.search(sentence)
        if m:
            claims.append(_check_presence(mask, metadata, m.group(0), m.group("cls")))
        else:
            m = _THERE_IS_RE.search(sentence)
            if m:
                claims.append(_check_presence(mask, metadata, m.group(0), m.group("cls")))

    # finditer, not search: tools/caption.py's own adjacency sentence packs
    # every pair into one "; "-joined clause ("Adjacent classes: A borders
    # B; C borders D; ..."), so a single first-match would silently check
    # only the first pair and ignore the rest.
    for m in _ADJACENCY_RE.finditer(sentence):
        claims.append(_check_adjacency(mask, metadata, m.group(0), m.group("a"), m.group("b")))

    return claims


def verify_answer(
    mask: np.ndarray,
    metadata: dict,
    answer: str,
    *,
    on_unverifiable: Literal["flag", "drop"] = "flag",
) -> VerificationResult:
    """Split `answer` into sentences, re-test every recognised factual
    claim in each against `mask` via evidence/ops.py, and return the
    verified answer (unrecognised/failing claims flagged in place, or
    dropped entirely if `on_unverifiable="drop"`) plus the full per-claim
    report. A sentence containing no recognised claim at all is left
    completely unchanged -- it was never a checkable assertion."""
    sentences = [s for s in (s.strip() for s in _SENTENCE_SPLIT_RE.split(answer.strip())) if s]

    all_claims: list[Claim] = []
    kept_sentences: list[str] = []

    for sentence in sentences:
        claims = _claims_in_sentence(mask, metadata, sentence)
        all_claims.extend(claims)

        if not claims or all(c.passed for c in claims):
            kept_sentences.append(sentence)
        elif on_unverifiable == "flag":
            failed = [c for c in claims if not c.passed]
            reasons = "; ".join(c.reason for c in failed if c.reason)
            kept_sentences.append(f"{sentence} [UNVERIFIED: {reasons}]")
        # else "drop": the whole sentence is omitted

    return VerificationResult(
        original_answer=answer,
        verified_answer=" ".join(kept_sentences),
        claims=all_claims,
    )
