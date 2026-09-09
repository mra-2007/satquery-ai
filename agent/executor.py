"""Runs a validated agent.dsl.Plan against the real, deterministic tool
implementations in evidence/ops.py and evidence/change.py.

This module's job is wiring, in two parts: first, agent.guardrail's
capability check rewrites any step the image's own resolution can't
actually support (see apply_capability_guardrail's docstring); then each
tool name the (possibly rewritten) plan uses gets bound to its real
implementation, over this call's mask/metadata (and before/after, for
change), and handed to agent.dsl.execute() -- which is where "validate
first, run every step in order, build a trace" already lives (agent/dsl.py,
tested there). This module doesn't re-implement that; it applies the
guardrail before execution and summarizes each step's raw output into
something small and serialisable for the trace afterward.

Because agent.dsl.execute() takes no LLM input, and everything here is
plain Python wiring over a fixed mask/metadata, running the exact same
Plan JSON against the exact same rasters always reproduces the exact same
outputs and ExecutionTrace -- replayable without the LLM, per the deck.
"""

import functools
from dataclasses import dataclass
from typing import Any

import numpy as np

from agent.dsl import Plan, execute as _dsl_execute
from agent.guardrail import apply_capability_guardrail
from evidence import change, fusion, metadata as metadata_tool, ops
from evidence.schema import TraceStep
from tools import caption as caption_tool
from tools import conversational as conversational_tool
from tools import cross_modal as cross_modal_tool
from tools import grounding as grounding_tool
from tools import verifier as verifier_tool

# Tools backed directly by evidence/ops.py. Each has the shape
# fn(mask, ...permitted_parameters..., metadata) -- mask and metadata are
# call-level context, never LLM-controlled parameters (see agent/registry.py).
_OPS_TOOLS = ("count", "size", "presence", "adjacency")


class ExecutorError(RuntimeError):
    """Raised when a plan needs a raster (mask, or before/after) that this
    run() call didn't provide."""


def _summarize(output: Any) -> Any:
    """A compact, JSON-serialisable summary of a tool's raw output, for the
    trace. Scalars (the count/size/presence/adjacency outputs) pass
    through unchanged. A change.ChangeReport's full change-mask raster is
    reduced to a changed-pixel count instead of being embedded whole, so
    the trace stays small regardless of raster size. A caption.CaptionResult's
    dataclasses are reduced to plain dicts/tuples -- both the final
    (possibly Gemini-smoothed) caption AND the untouched, fact-only
    template_sentence are kept, so the trace itself is the audit trail
    proving Gemini didn't change a fact. A grounding.GroundingResult's
    dataclass and Polygon models are likewise reduced to plain dicts, as
    is a cross_modal.CrossModalResult's list of per-class findings."""
    if isinstance(output, change.ChangeReport):
        return {
            "changed_pixels": int(output.mask.sum()),
            "area_changes": output.area_changes,
            "summary": output.summary,
        }
    if isinstance(output, caption_tool.CaptionResult):
        return {
            "caption": output.caption,
            "template_sentence": output.template_sentence,
            "smoothed": output.smoothed,
            "facts": [
                {"class_id": f.class_id, "class_name": f.class_name,
                 "area_ha": f.area_ha, "fragment_count": f.fragment_count}
                for f in output.facts
            ],
            "adjacent_pairs": output.adjacent_pairs,
        }
    if isinstance(output, grounding_tool.GroundingResult):
        return {
            "class_id": output.class_id,
            "class_name": output.class_name,
            "area_ha": output.area_ha,
            "candidate_count": output.candidate_count,
            "qualifier": output.qualifier,
            "centroid_pixel": output.centroid_pixel,
            "centroid_lonlat": output.centroid_lonlat,
            "pixel_bbox": output.pixel_bbox,
            "geo_bbox": output.geo_bbox,
            "pixel_polygon": output.pixel_polygon.model_dump(),
            "geo_polygon": output.geo_polygon.model_dump() if output.geo_polygon is not None else None,
        }
    if isinstance(output, cross_modal_tool.CrossModalResult):
        return {
            "cloud_simulated": output.cloud_simulated,
            "cloud_fraction": output.cloud_fraction,
            "summary": output.summary,
            "sensor_status": output.sensor_status,
            "findings": [
                {"class_id": f.class_id, "class_name": f.class_name,
                 "optical_area_ha": f.optical_area_ha, "sar_area_ha": f.sar_area_ha,
                 "fused_area_ha": f.fused_area_ha, "detected_by": f.detected_by,
                 "attribution": f.attribution}
                for f in output.findings
            ],
        }
    if isinstance(output, fusion.FusionResult):
        return {
            "class_id": output.class_id,
            "class_name": output.class_name,
            "fused_present": output.fused_present,
            "fused_confidence": output.fused_confidence,
            "agreement": output.agreement,
            "agreement_fraction": output.agreement_fraction,
            "correlated_sources": list(output.correlated_sources),
            "weighting_note": output.weighting_note,
            "summary": output.summary,
            "sources": [
                {"source": s.source, "available": s.available, "present": s.present,
                 "confidence": s.confidence, "weight": s.weight, "detail": s.detail}
                for s in output.sources
            ],
        }
    if isinstance(output, metadata_tool.MetadataResult):
        return {"aspect": output.aspect, "answer_text": output.answer_text, "detail": output.detail}
    if isinstance(output, conversational_tool.ConversationalReply):
        return {"kind": output.kind, "reply": output.reply}
    if isinstance(output, verifier_tool.VerificationResult):
        return {
            "original_answer": output.original_answer,
            "verified_answer": output.verified_answer,
            "all_passed": output.all_passed,
            "claims": [
                {"text": c.text, "claim_type": c.claim_type, "tool": c.tool, "class_name": c.class_name,
                 "claimed_value": c.claimed_value, "recomputed_value": c.recomputed_value,
                 "passed": c.passed, "reason": c.reason}
                for c in output.claims
            ],
        }
    return output


@dataclass
class ExecutionTrace:
    """Every step's task, tool name, parameters, output summary, and
    confidence, in call order -- plain data, so it can be logged, diffed,
    or handed to someone else to replay, with no LLM involved."""

    steps: list[TraceStep]

    def __iter__(self):
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.steps)


def run(
    plan: Plan,
    metadata: dict,
    *,
    mask: np.ndarray | None = None,
    before: np.ndarray | None = None,
    after: np.ndarray | None = None,
    stack: np.ndarray | None = None,
    classes: dict[int, str] | None = None,
) -> tuple[dict[str, Any], ExecutionTrace]:
    """Run every step of `plan` against evidence/ops.py
    (count/size/presence/adjacency, over `mask`), evidence/change.py
    (change, over `before`/`after`), tools/caption.py and tools/grounding.py
    (caption/ground, over `mask`), tools/cross_modal.py (cross_modal,
    over the raw `stack` -- it segments internally, three ways, so it
    needs the model's full multi-channel input, not a pre-computed mask),
    evidence/fusion.py (fusion, also over the raw `stack` -- it needs
    the real optical bands for its spectral index and the real SAR
    channels when present, neither of which survive in a pre-computed mask),
    evidence/metadata.py (metadata, fact retrieval from the scene's own
    record -- `metadata['filename']`/`metadata['sensor']` when the caller
    has them), and tools/conversational.py (conversational, a canned
    greeting/off-topic reply needing no raster at all).

    When `classes` (the scene's class_id -> name mapping) is given, the
    capability guardrail runs first: any `count` step targeting a class
    that isn't resolvable at `metadata['gsd_metres']` is rewritten to a
    `size` step, and the trace records why. Without `classes` there's no
    way to know what a class_id represents, so the guardrail is skipped --
    pass it whenever you have it.

    Returns (result, trace): `result` maps each step's id to its full,
    unsummarized output (e.g. the actual change.ChangeReport, not a
    summary of it); `trace` is the ExecutionTrace described above.

    Raises agent.dsl.PlanValidationError if the plan itself is invalid
    (unknown tool, out-of-schema parameters, duplicate ids) or references a
    tool with no implementation here (e.g. segment -- not backed by
    evidence/ops.py, evidence/change.py, tools/caption.py, tools/grounding.py,
    or tools/cross_modal.py). Raises ExecutorError if the plan needs a
    raster this call didn't provide.
    """
    guardrail_trace: list[TraceStep] = []
    if classes is not None:
        plan, guardrail_trace = apply_capability_guardrail(plan, classes, metadata["gsd_metres"])

    needed_tools = {step.tool for step in plan}
    tool_functions: dict[str, Any] = {}

    if needed_tools & set(_OPS_TOOLS):
        if mask is None:
            raise ExecutorError(
                "plan uses an evidence.ops tool (count/size/presence/adjacency) "
                "but no mask was provided"
            )
        for name in _OPS_TOOLS:
            tool_functions[name] = functools.partial(getattr(ops, name), mask, metadata=metadata)

    if "change" in needed_tools:
        if before is None or after is None:
            raise ExecutorError(
                "plan uses the 'change' tool but before and after rasters were not both provided"
            )
        # class_names=classes so ChangeReport.summary reads "Urban fabric
        # gained ..." rather than "class 0 gained ..." -- omitted only when
        # the caller has no class mapping at all (classes=None), the same
        # condition that already skips the capability guardrail above.
        tool_functions["change"] = functools.partial(
            change.detect_change, before, after, metadata, class_names=classes,
        )

    if "caption" in needed_tools:
        if mask is None:
            raise ExecutorError("plan uses the 'caption' tool but no mask was provided")
        tool_functions["caption"] = functools.partial(caption_tool.caption, mask, metadata)

    if "ground" in needed_tools:
        if mask is None:
            raise ExecutorError("plan uses the 'ground' tool but no mask was provided")
        tool_functions["ground"] = functools.partial(grounding_tool.ground, mask, metadata)

    if "cross_modal" in needed_tools:
        if stack is None:
            raise ExecutorError("plan uses the 'cross_modal' tool but no stack was provided")
        tool_functions["cross_modal"] = functools.partial(cross_modal_tool.cross_modal_analysis, stack, metadata)

    if "verify" in needed_tools:
        if mask is None:
            raise ExecutorError("plan uses the 'verify' tool but no mask was provided")
        tool_functions["verify"] = functools.partial(verifier_tool.verify_answer, mask, metadata)

    if "fusion" in needed_tools:
        if stack is None:
            raise ExecutorError("plan uses the 'fusion' tool but no stack was provided")
        tool_functions["fusion"] = functools.partial(fusion.fuse_evidence, stack, metadata)

    if "metadata" in needed_tools:
        if mask is None:
            raise ExecutorError("plan uses the 'metadata' tool but no mask was provided")
        tool_functions["metadata"] = functools.partial(metadata_tool.scene_metadata, mask, metadata)

    if "conversational" in needed_tools:
        # No raster dependency at all -- a greeting or an off-topic
        # question isn't about the scene, so there is nothing to bind.
        tool_functions["conversational"] = conversational_tool.reply

    dsl_result = _dsl_execute(plan, tool_functions)

    trace_steps = guardrail_trace + [
        step.model_copy(update={"output": _summarize(step.output)})
        for step in dsl_result.trace
    ]

    # Per CLAUDE.md's "No pixel, no claim": tools/caption.py's `caption`
    # field is the one place in this codebase where text can genuinely
    # drift from the mask -- it's Gemini's reworded version of a fact-only
    # template, and although caption.py's prompt tells Gemini not to change
    # any fact, that instruction isn't a guarantee. Every caption step is
    # therefore automatically re-checked here, independent of whatever the
    # plan itself asked for, so a verification report always appears in the
    # trace whenever a caption was produced -- the LLM never has to think
    # to ask for it. Uses `tool="verify"` (not a new tool name) so a caller
    # can find any verification step, automatic or explicitly planned, the
    # same way: `step.tool == "verify"`.
    if mask is not None:
        for step in plan:
            if step.tool != "caption" or step.id not in dsl_result.outputs:
                continue
            caption_result = dsl_result.outputs[step.id]
            verification = verifier_tool.verify_answer(mask, metadata, caption_result.caption)
            trace_steps.append(TraceStep(
                task=f"verify:{step.id}",
                tool="verify",
                parameters={"answer": caption_result.caption},
                output=_summarize(verification),
                confidence=1.0,
            ))

    return dsl_result.outputs, ExecutionTrace(trace_steps)
