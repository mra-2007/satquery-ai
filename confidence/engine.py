"""Produces a SINGLE calibrated confidence number for one answer, from
five independent signals:

1. **Perception margin** -- the model's own temperature-calibrated
   softmax margin (top-1 minus top-2 probability) over the pixels that
   actually make up the answer. Wide margin = the model wasn't torn
   between two classes there.
2. **TTA stability** -- the SAME scene, segmented 8 times under the 8
   elements of the dihedral group D4 (0/90/180/270 degree rotations, each
   with and without a horizontal flip -- "8 flips/rotations"), each
   prediction un-rotated back to the original orientation before
   comparing. A real feature should look roughly the same size regardless
   of which way the image happened to be oriented; high variance in the
   answer class's measured area across the 8 means the model is reacting
   to orientation, not content -- a red flag no single forward pass alone
   can catch.
3. **Cross-source agreement** -- evidence/fusion.py's independent
   corroboration (segmentation, classification, a spectral index, SAR
   when available) for the answer's class, when the raw stack is at hand
   to compute it.
4. **Resolution suitability** -- agent/registry.py's capability table:
   is the answer class's typical real-world size even meaningfully
   resolvable at this image's GSD, or is the model being asked to measure
   something close to or below its physical resolution limit?
5. **Plan validity** -- did agent/planner.py's plan_from_query() get a
   valid plan out of Gemini on the first try, or did it need one or more
   retries (agent/planner.py's own retry-with-feedback loop)? A plan that
   needed correction suggests the question itself was more ambiguous or
   harder to translate into a tool call, independent of the image.

Per CLAUDE.md's "No pixel, no claim": DETERMINISTIC geometry (count/size/
presence/adjacency/intersect's own arithmetic over an ALREADY-COMPUTED
mask, in evidence/ops.py) is exact -- given a mask, there is no
uncertainty left in summing its pixels. That is DETERMINISTIC_GEOMETRY_
CONFIDENCE below (always 1.0, unconditionally) -- it is reported
ALONGSIDE the calibrated number this module produces, never multiplied
into it and never replacing it: "1.0 conditional on the mask" and "how
much should you trust the mask itself" are two different, both honest,
numbers, and a caller (api/rendering.py, agent/executor.py) should surface
both rather than collapsing them into one figure that hides which claim
it's making.

Missing signals degrade gracefully, not silently
------------------------------------------------------
TTA stability and cross-source agreement both need the scene's raw
multi-channel stack and a loaded model session -- not always at hand
(e.g. only a saved mask, no stack, is available). When a signal can't be
computed, ComputeConfidence excludes it from the weighted average and
renormalizes the remaining weights to still sum to 1, rather than
silently substituting a fake neutral value that would quietly change the
score's meaning. `ConfidenceResult.components` records exactly which
signals contributed and which were skipped (and why), so this is always
inspectable, not just trusted.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agent.registry import capability_table
from confidence.calibration import load_temperature, temperature_scale
from evidence.ops import ClassId, class_mask

ClassIdOrNone = ClassId | None


# --- Deterministic geometry: exact, conditional on the mask -----------------

DETERMINISTIC_GEOMETRY_CONFIDENCE = 1.0
DETERMINISTIC_GEOMETRY_SOURCE = "evidence.ops (deterministic geometry, conditional on mask)"


# --- Component 1: perception margin ------------------------------------------


def perception_margin(probabilities: np.ndarray, class_id: ClassId, *, temperature: float | None = None) -> float:
    """Mean (top-1 minus top-2 probability) over every pixel whose argmax
    class is one of `class_id` -- i.e. over the pixels that actually
    produced the answer, not the whole scene. `probabilities` is
    temperature-scaled first (confidence/calibration.py's fitted T,
    loaded fresh if not passed) -- this is the one place raw model output
    would otherwise leak through as a number, which is exactly what
    CLAUDE.md's "must never be reported as a percentage" forbids.

    A MEAN over the answer's own footprint, not the PEAK evidence/fusion.py
    uses for its existence check -- this is a "how sure is this whole
    measurement" question, not "does at least one confident pixel of this
    class exist anywhere" one, so the spatial aggregate has to match: one
    lucky confident pixel in an otherwise ambiguous region should not
    report high confidence for the region's own size/count/adjacency.

    Returns 0.5 (genuinely uninformative -- neither confident nor not) if
    `class_id` has no pixels in the mask at all: there is no margin to
    measure for a class the model didn't predict anywhere.
    """
    if temperature is None:
        temperature = load_temperature()
    calibrated = temperature_scale(probabilities, temperature)

    mask = np.argmax(calibrated, axis=0)
    answer_pixels = class_mask(mask, class_id)
    if not np.any(answer_pixels):
        return 0.5

    sorted_probs = np.sort(calibrated[:, answer_pixels], axis=0)  # ascending along the class axis
    top1 = sorted_probs[-1, :]
    top2 = sorted_probs[-2, :]
    return float(np.mean(top1 - top2))


# --- Component 2: TTA stability ----------------------------------------------

# The 8 elements of D4 (the square's symmetry group): (k, flip) means
# "rotate 90*k degrees counterclockwise, then flip horizontally iff flip".
# Applying the SAME element's inverse to a prediction made on the
# transformed input undoes it exactly, so all 8 predictions can be
# compared in the original image's own orientation.
_D4_ELEMENTS: list[tuple[int, bool]] = [(k, flip) for flip in (False, True) for k in range(4)]


def _apply_d4(stack: np.ndarray, k: int, flip: bool) -> np.ndarray:
    """Rotate (spatial axes 1, 2) by 90*k degrees counterclockwise, then
    flip horizontally (axis 2) iff `flip`."""
    transformed = np.rot90(stack, k=k, axes=(1, 2))
    if flip:
        transformed = np.flip(transformed, axis=2)
    return np.ascontiguousarray(transformed)


def _invert_d4(array: np.ndarray, k: int, flip: bool, *, spatial_axes: tuple[int, int]) -> np.ndarray:
    """Undo _apply_d4 on a (..., H, W)-shaped prediction (probabilities or
    mask) made on the transformed input, so it lines back up with the
    ORIGINAL image's pixel grid. Order matters: undo the flip first, then
    the rotation -- the exact reverse of how they were applied."""
    result = array
    if flip:
        result = np.flip(result, axis=spatial_axes[1])
    result = np.rot90(result, k=-k, axes=spatial_axes)
    return np.ascontiguousarray(result)


def tta_stability(
    stack: np.ndarray,
    sensor: str,
    class_id: ClassId,
    metadata: dict,
    *,
    session: Any,
    config: Any,
    max_relative_std: float = 0.15,
) -> float:
    """Segments `stack` under all 8 D4 transforms ("8 flips/rotations"),
    measures `class_id`'s area FRACTION of the scene under each (after
    undoing the transform so every measurement is of the same real
    pixels), and converts the spread across those 8 measurements into a
    [0, 1] stability score: `1 - std/mean(area_fraction) / max_relative_std`,
    clamped to [0, 1] -- a coefficient of variation, so a genuinely tiny
    feature isn't penalized on the same ABSOLUTE std scale as a large one
    (a std of 0.01 is huge instability for a class covering 2% of the
    scene, and negligible noise for one covering 60% of it).

    `max_relative_std` (0.15, disclosed and documented, not fit to any
    benchmark) is the coefficient of variation above which stability is
    scored 0 -- a feature whose measured extent swings by 15%+ of its own
    mean size purely from being rotated/flipped is not a measurement to
    trust as-is.

    Costs 8 real forward passes (perception.infer.segment_image, one per
    transform) -- ~0.7s for a single 120x120 tile on CPU, measured during
    this module's own development; scales with image size the same way
    any other segment_image() call does, since it reuses the exact same
    tiling/stitching pipeline, just called 8 times.
    """
    from perception import infer  # deferred: keeps a plain import of this module cheap when TTA isn't used

    area_fractions: list[float] = []
    for k, flip in _D4_ELEMENTS:
        transformed_stack = _apply_d4(stack, k, flip)
        result = infer.segment_image(transformed_stack, sensor, session=session, config=config)
        realigned_mask = _invert_d4(result.mask, k, flip, spatial_axes=(0, 1))
        answer_pixels = class_mask(realigned_mask, class_id)
        area_fractions.append(float(np.mean(answer_pixels)))

    fractions = np.asarray(area_fractions, dtype=np.float64)
    mean_fraction = float(np.mean(fractions))
    if mean_fraction <= 0.0:
        # The class was never detected under ANY of the 8 orientations --
        # perfectly stable absence, not an unstable/undefined measurement.
        return 1.0

    coefficient_of_variation = float(np.std(fractions)) / mean_fraction
    return float(np.clip(1.0 - coefficient_of_variation / max_relative_std, 0.0, 1.0))


# --- Component 3: cross-source agreement -------------------------------------


def cross_source_agreement(stack: np.ndarray, metadata: dict, class_id: ClassId, *, session: Any, config: Any) -> float:
    """evidence/fusion.py's own agreement_fraction for `class_id` --
    independent corroboration from up to 4 genuinely different sources
    (segmentation, classification, a spectral index, SAR when real SAR is
    present). See evidence/fusion.py's own module docstring for why these
    four, and why segmentation+classification count as correlated rather
    than fully independent evidence."""
    from evidence import fusion

    result = fusion.fuse_evidence(stack, metadata, class_id, session=session, config=config)
    return result.agreement_fraction


# --- Component 4: resolution suitability -------------------------------------


def resolution_suitability(class_id: ClassIdOrNone, classes: dict[int, str] | None, metadata: dict) -> float:
    """How comfortably `class_id`'s typical real-world size clears this
    image's minimum resolvable object size (agent.registry.capability_table
    -- min_resolvable_m = 2.5 * gsd_metres, per CLAUDE.md).

    score = clip(ratio / 2, 0, 1), where ratio = typical_object_size_m /
    min_resolvable_m: ratio=1.0 (right at the guardrail's own countability
    boundary) scores 0.5; ratio=2.0 (comfortably twice the minimum
    resolvable size) scores a full 1.0; ratio=0.5 (half the minimum, badly
    sub-resolution) scores 0.25. Smooth and monotonic, so a class just
    over the boundary isn't treated identically to one comfortably clear
    of it, unlike the guardrail's own strict yes/no countable check.

    Classes with no entry in agent.registry.TYPICAL_OBJECT_SIZE_M (most of
    the real 19-class vocabulary -- bulk land-cover types like "Arable
    land" or "Inland waters" measured as area, not counted as discrete
    objects, so this ratio doesn't apply to them at all) and list class_ids
    (a generic noun resolved to several real classes at once -- see
    agent/vocabulary.py) both default to 1.0: no evidence of a resolution
    problem, not proof of full suitability, but there is nothing more
    specific to say about them.
    """
    if class_id is None or isinstance(class_id, list) or classes is None:
        return 1.0
    class_name = classes.get(class_id)
    if class_name is None:
        return 1.0

    table = capability_table(metadata)
    capability = table.classes.get(class_name)
    if capability is None:
        return 1.0

    ratio = capability.typical_object_size_m / capability.min_resolvable_m
    return float(np.clip(ratio / 2.0, 0.0, 1.0))


# --- Component 5: plan validity -----------------------------------------------

PLAN_VALIDITY_DECAY_PER_RETRY = 0.2
PLAN_VALIDITY_FLOOR = 0.5


def plan_validity(attempts: int) -> float:
    """1.0 if agent/planner.py's plan_from_query() got a valid plan out of
    Gemini on the FIRST attempt; each retry (Gemini's own response failed
    agent.dsl.validate() and had to be re-asked with the error fed back)
    costs PLAN_VALIDITY_DECAY_PER_RETRY, down to a floor of
    PLAN_VALIDITY_FLOOR -- a plan that needed correction is still a VALID
    plan by the time execution sees it (agent.dsl.validate() already
    guarantees that), but the retry itself is real evidence the question
    was harder to translate into a tool call than a typical one.

    `attempts` is meaningless (and this returns 1.0 unconditionally) for a
    cache hit or a keyword-fallback plan -- neither one retried anything;
    see agent/planner.py's own plan_from_query() for how a caller
    determines `attempts` in the first place.
    """
    if attempts <= 1:
        return 1.0
    return max(PLAN_VALIDITY_FLOOR, 1.0 - PLAN_VALIDITY_DECAY_PER_RETRY * (attempts - 1))


# --- The refusal threshold: set empirically, not at a hardcoded 90% --------

# Per eval/RESULTS.md (eval/run_benchmark.py's own real, disclosed
# benchmark -- 654 real BigEarthNet.txt questions scored against the
# PREDICTED mask, i.e. what this system actually answers with in
# production, not the ground-truth-mask ceiling table):
#
#     | Category  | Accuracy |
#     |-----------|----------|
#     | presence  |   72.6%  |
#     | count     |   44.0%  |
#     | area      |   76.1%  |
#     | adjacency |   60.6%  |
#     | Overall   |   62.7%  |
#
# and models/satquery_model.onnx's own reported training-set mIoU is
# 0.4654. A system whose real, measured overall accuracy is 62.7% -- and
# whose weakest routinely-attempted category (count, dragged down by
# segmentation noise fragmenting/merging regions) sits at 44.0% -- cannot
# honestly demand 90% confidence before it will answer: at that bar,
# nearly EVERY answer would be refused, correct ones included, which
# defeats the purpose of a confidence score entirely (a threshold no real
# output can ever clear is not a threshold, it is a refusal to function).
#
# REFUSAL_THRESHOLD = 0.55 instead: comfortably BELOW the 62.7% overall
# average (so a "typical", average-trustworthiness answer is NOT refused
# just for being average) and comfortably ABOVE the weakest category's
# 44.0% (so an answer whose components genuinely look as unreliable as a
# typical "count" question -- exactly the category this system is most
# often wrong about -- IS refused, rather than confidently reported
# anyway). It is deliberately not set exactly AT 62.7%: with only 654
# benchmark questions behind it and no confidence-vs-correctness curve yet
# measured for THIS specific engine (that would need per-question
# confidence scores paired with correctness over a real benchmark run,
# which this initial version doesn't yet have), a conservative buffer
# below the measured average avoids over-refusing on calibration noise
# alone while still enforcing a real, reachable, evidence-grounded bar --
# not the unreachable 90% this section replaces.
REFUSAL_THRESHOLD = 0.55


def should_refuse(calibrated_confidence: float) -> bool:
    return calibrated_confidence < REFUSAL_THRESHOLD


# --- Combining everything into one calibrated number -------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "perception_margin": 0.30,
    "tta_stability": 0.20,
    "cross_source_agreement": 0.20,
    "resolution_suitability": 0.15,
    "plan_validity": 0.15,
}
assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9


@dataclass
class ConfidenceResult:
    calibrated: float
    components: dict[str, float | None] = field(default_factory=dict)
    weights_used: dict[str, float] = field(default_factory=dict)
    geometry_confidence: float = DETERMINISTIC_GEOMETRY_CONFIDENCE
    refused: bool = False


def combine(components: dict[str, float | None], weights: dict[str, float] | None = None) -> ConfidenceResult:
    """Weighted average of whichever of DEFAULT_WEIGHTS' five components
    are actually present (not None) in `components`, with the missing
    ones' weight redistributed proportionally across the rest -- so a
    genuinely unavailable signal (no stack for TTA/agreement, no
    plan-attempt count) never silently counts as a neutral 0.5 or a
    perfect 1.0, it is simply excluded and the remaining signals are
    reweighted to still sum to 1.
    """
    weights = weights or DEFAULT_WEIGHTS
    available = {name: value for name, value in components.items() if value is not None and name in weights}
    if not available:
        raise ValueError("no confidence components available to combine")

    total_weight = sum(weights[name] for name in available)
    weights_used = {name: weights[name] / total_weight for name in available}
    calibrated = sum(weights_used[name] * available[name] for name in available)

    return ConfidenceResult(
        calibrated=float(np.clip(calibrated, 0.0, 1.0)),
        components=dict(components),
        weights_used=weights_used,
        refused=should_refuse(calibrated),
    )


@dataclass
class ConfidenceInputs:
    """Everything compute_confidence() can use, all optional except
    `class_id` -- callers pass whatever they actually have; unavailable
    fields simply drop that signal (see combine()'s own docstring)."""

    class_id: ClassId
    metadata: dict
    classes: dict[int, str] | None = None
    probabilities: np.ndarray | None = None       # for perception_margin
    stack: np.ndarray | None = None                # for tta_stability / cross_source_agreement
    sensor: str = "fused"
    session: Any = None
    config: Any = None
    plan_attempts: int | None = None
    run_tta: bool = True
    run_agreement: bool = True


def compute_confidence(inputs: ConfidenceInputs, weights: dict[str, float] | None = None) -> ConfidenceResult:
    """The main entry point: computes whichever of the five components
    `inputs` supports, combines them via combine() above, and returns the
    single calibrated ConfidenceResult -- this is what
    agent/executor.py wires into the trace, and what api/rendering.py
    turns into an extra evidence.schema.SourceConfidence entry alongside
    the unchanged, always-1.0 deterministic-geometry one."""
    components: dict[str, float | None] = {
        "perception_margin": None,
        "tta_stability": None,
        "cross_source_agreement": None,
        "resolution_suitability": resolution_suitability(inputs.class_id, inputs.classes, inputs.metadata),
        "plan_validity": plan_validity(inputs.plan_attempts) if inputs.plan_attempts is not None else None,
    }

    if inputs.probabilities is not None:
        components["perception_margin"] = perception_margin(inputs.probabilities, inputs.class_id)

    if inputs.stack is not None and inputs.session is not None and inputs.config is not None:
        if inputs.run_tta:
            components["tta_stability"] = tta_stability(
                inputs.stack, inputs.sensor, inputs.class_id, inputs.metadata,
                session=inputs.session, config=inputs.config,
            )
        if inputs.run_agreement:
            components["cross_source_agreement"] = cross_source_agreement(
                inputs.stack, inputs.metadata, inputs.class_id, session=inputs.session, config=inputs.config,
            )

    return combine(components, weights)
