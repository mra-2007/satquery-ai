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

    if tool in ("count", "size", "presence"):
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

    return _whole_scene_geometry(mask)  # caption, cross_modal, change


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
    if tool in ("cross_modal", "change"):
        return str(output["summary"]), None, str(output["summary"]), limitation
    return str(output), None, str(output), limitation


def build_confidence(classification: ClassificationResult, full_trace: list[TraceStep]) -> list[SourceConfidence]:
    """One entry for the task classification, one for the final tool call
    -- every number here already existed in the trace; this just repacks
    it into evidence.schema's confidence contract."""
    confidences = [SourceConfidence(source="agent.tasks.classify_task", confidence=classification.confidence)]
    execute_steps = [step for step in full_trace if step.task.startswith("execute:")]
    if execute_steps:
        final_step = execute_steps[-1]
        confidences.append(SourceConfidence(source=f"tool:{final_step.tool}", confidence=final_step.confidence))
    return confidences
