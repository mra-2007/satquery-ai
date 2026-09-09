"""Turns an already-computed execution trace into the pieces
evidence.schema.Evidence needs: value, units, geometry, and confidence.

Per CLAUDE.md's "No pixel, no claim": every one of these is read off the
trace or vectorized straight from the mask (rasterio.features.shapes,
the same approach tools/grounding.py uses) -- nothing here is invented or
asked of an LLM. This module only formats what agent/executor.py and
evidence/ops.py already computed.
"""

import numpy as np
from rasterio import features
from rasterio.transform import Affine

from agent.tasks import ClassificationResult
from confidence import engine as confidence_engine
from evidence.ops import ClassId, class_mask
from evidence.schema import Feature, FeatureCollection, Polygon, SourceConfidence, TraceStep


def _class_label(classes: dict[int, str], class_id: ClassId) -> str:
    """A human-readable label for a class_id that might be a list (a
    generic noun like "forest" resolved to several real classes at once --
    see agent/vocabulary.py and evidence/ops.py)."""
    if isinstance(class_id, list):
        return " / ".join(classes.get(c, str(c)) for c in class_id)
    return classes.get(class_id, str(class_id))


def _vectorize_class(mask: np.ndarray, class_id: ClassId, class_name: str) -> list[Feature]:
    """Every connected region of `class_id` (or the union of a list of
    class_ids) in `mask`, as GeoJSON Features in pixel coordinates -- the
    actual pixels responsible for the answer, not an invented geometry."""
    binary = class_mask(mask, class_id)
    if not binary.any():
        return []
    shapes = features.shapes(binary.astype(np.uint8), mask=binary, transform=Affine.identity())
    return [
        Feature(geometry=Polygon.model_validate(geom), properties={"class": class_name})
        for geom, value in shapes
        if value == 1
    ]


def _whole_scene_geometry(mask: np.ndarray) -> FeatureCollection:
    """Fallback geometry for tools with no single driving class_id
    (caption/cross_modal/change): the scene's own pixel extent, so
    Evidence.geometry is never fabricated content, just a plain bounding
    box of what was actually analysed."""
    height, width = mask.shape
    ring = [[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)], [0.0, 0.0]]
    return FeatureCollection(
        features=[Feature(geometry=Polygon(coordinates=[ring]), properties={"note": "whole scene extent"})]
    )


def build_geometry(mask: np.ndarray, classes: dict[int, str], final_step: TraceStep) -> FeatureCollection:
    """The GeoJSON backing `final_step`'s answer -- which class(es)'
    pixels actually produced it."""
    tool, params, output = final_step.tool, final_step.parameters, final_step.output

    if tool in ("count", "size", "presence", "fusion"):
        class_id = params["class_id"]
        return FeatureCollection(features=_vectorize_class(mask, class_id, _class_label(classes, class_id)))

    if tool == "adjacency":
        class_a, class_b = params["class_a"], params["class_b"]
        return FeatureCollection(features=(
            _vectorize_class(mask, class_a, _class_label(classes, class_a))
            + _vectorize_class(mask, class_b, _class_label(classes, class_b))
        ))

    if tool == "ground":
        # final_step.output is already agent.executor._summarize()'s plain-dict
        # summary of a grounding.GroundingResult -- prefer the geographic
        # polygon when one was computed, else the pixel polygon.
        polygon = output.get("geo_polygon") or output["pixel_polygon"]
        return FeatureCollection(features=[
            Feature(geometry=Polygon.model_validate(polygon), properties={"class": output["class_name"]})
        ])

    # caption, cross_modal, change, intersect: 'intersect' would need BOTH
    # the before and after rasters to vectorize the true overlap region, but
    # this function only receives the one `mask` the rest of these tools
    # already share (the current/after mask) -- same reason 'change' itself
    # falls back to the whole scene extent rather than a precise geometry.
    return _whole_scene_geometry(mask)


def _pluralize(class_name: str) -> str:
    """Several of agent.vocabulary.SEGMENTATION_CLASSES's real names are
    already plural ("Inland waters", "Permanent crops") -- appending
    "(s)" unconditionally would read as "inland waters(s)"."""
    return class_name if class_name.endswith("s") else f"{class_name}(s)"


def build_answer(full_trace: list[TraceStep], classes: dict[int, str]):
    """Returns (value, units, answer_text, limitation) -- value/units feed
    Evidence directly; answer_text is the same human-readable rendering
    scripts/demo_real.py's render_answer() produces; limitation is the
    capability guardrail's message, or None if it never fired."""
    limitation = next(
        (step.output["limitation"] for step in full_trace if step.task == "capability_guardrail"), None
    )

    execute_steps = [step for step in full_trace if step.task.startswith("execute:")]
    if not execute_steps:
        raise ValueError("no execute step found in trace")
    final_step = execute_steps[-1]
    tool, output, params = final_step.tool, final_step.output, final_step.parameters

    if tool == "count":
        class_name = _pluralize(_class_label(classes, params["class_id"]).lower())
        return int(output), class_name, f"{output} {class_name}", limitation
    if tool == "size":
        class_name = _class_label(classes, params["class_id"]).lower()
        return float(output), "hectares", f"{output:.2f} ha of {class_name}", limitation
    if tool in ("presence", "adjacency"):
        return bool(output), None, ("yes" if output else "no"), limitation
    if tool == "caption":
        return str(output["caption"]), None, str(output["caption"]), limitation
    if tool == "ground":
        text = f"{output['class_name']} at pixel bbox {tuple(output['pixel_bbox'])}"
        return str(output["class_name"]), None, text, limitation
    if tool == "cross_modal":
        return str(output["summary"]), None, str(output["summary"]), limitation
    if tool == "change":
        # A focused change question ("how much land changed to water?")
        # carries its own scalar answer alongside the full summary -- see
        # evidence/change.py's ChangeReport.focus_gained_ha. An unfocused
        # "what changed?" has no class_id parameter at all, so it keeps
        # returning the plain-English summary as before.
        if output.get("focus_gained_ha") is not None:
            class_name = _class_label(classes, params["class_id"]).lower()
            gained = output["focus_gained_ha"]
            text = f"{gained:.2f} ha changed to {class_name}. {output['summary']}"
            return float(gained), "hectares", text, limitation
        return str(output["summary"]), None, str(output["summary"]), limitation
    if tool == "intersect":
        # "how much cropland is flooded?" -- class_a is matched in the
        # before date, class_b in the after date, at the same pixels (see
        # evidence/ops.py's intersect_area).
        class_a_name = _class_label(classes, params["class_a"]).lower()
        class_b_name = _class_label(classes, params["class_b"]).lower()
        text = f"{output:.2f} ha of {class_a_name} is now {class_b_name}"
        return float(output), "hectares", text, limitation
    if tool == "fusion":
        return bool(output["fused_present"]), None, str(output["summary"]), limitation
    if tool in ("metadata", "conversational"):
        # Both are prose fact-retrieval/canned replies -- the full sentence
        # IS the answer, not a number a units label would attach to (see
        # evidence/metadata.py's/tools/conversational.py's own docstrings).
        text = str(output["answer_text"] if tool == "metadata" else output["reply"])
        return text, None, text, limitation
    return str(output), None, str(output), limitation


def build_confidence(classification: ClassificationResult, full_trace: list[TraceStep]) -> list[SourceConfidence]:
    """One entry for the task classification, one for the final tool call
    -- every number here already existed in the trace; this just repacks
    it into evidence.schema's confidence contract.

    When agent/executor.py ran confidence/engine.py for the final step
    (recorded as a separate `confidence:{step_id}` trace entry -- see
    executor.run()'s own docstring), two more entries are added: the
    engine's single calibrated number (already, via that same executor
    wiring, what `tool:{final_step.tool}` above reports too -- this entry
    just names its real source explicitly) and confidence/engine.py's own
    DETERMINISTIC_GEOMETRY_CONFIDENCE (always 1.0), so "the mask's own
    geometry arithmetic is exact" and "how much to trust the mask itself"
    are always two separately labelled, never conflated, numbers."""
    confidences = [SourceConfidence(source="agent.tasks.classify_task", confidence=classification.confidence)]
    execute_steps = [step for step in full_trace if step.task.startswith("execute:")]
    if execute_steps:
        final_step = execute_steps[-1]
        confidences.append(SourceConfidence(source=f"tool:{final_step.tool}", confidence=final_step.confidence))

        final_step_id = final_step.task.split(":", 1)[1]
        confidence_step = next(
            (step for step in full_trace if step.task == f"confidence:{final_step_id}"), None
        )
        if confidence_step is not None:
            confidences.append(SourceConfidence(
                source="confidence_engine.calibrated", confidence=confidence_step.output["calibrated"],
            ))
            confidences.append(SourceConfidence(
                source=confidence_engine.DETERMINISTIC_GEOMETRY_SOURCE,
                confidence=confidence_step.output["geometry_confidence"],
            ))
    return confidences
