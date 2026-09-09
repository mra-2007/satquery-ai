"""The agent's tool registry: what tools exist, which images they apply to,
what parameters they accept, and which land-cover classes are countable in
a given image.

Per CLAUDE.md: the LLM only classifies the task and emits a plan of tool
calls -- it never invents numbers. This module is the guardrail around that
plan: `select_model` decides which tools are even usable on a given image
(by GSD and modality), `validate_parameters` rejects any tool call whose
parameters don't match that tool's declared schema, and `capability_table`
decides which classes can honestly be counted at this image's resolution.
Every one of those decisions is appended to a returned execution trace
(evidence.schema.TraceStep), so nothing here is a silent judgment call.
"""

from dataclasses import dataclass
from typing import Any, Literal

from evidence.schema import TraceStep

Modality = Literal["optical", "sar", "both"]


class ParameterValidationError(ValueError):
    """Raised by validate_parameters() when a tool call's parameters don't
    match that tool's permitted schema."""


class _ClassIdOrList:
    """Sentinel schema type for a class_id/class_a/class_b parameter:
    either a single int, or a non-empty list[int]. Needed because
    agent.vocabulary.resolve_noun() can resolve a generic noun ("forest",
    "water") to several real classes at once rather than picking one
    arbitrarily -- evidence/ops.py's tools union a list into one mask
    before measuring, per evidence/ops.py's own docstring."""

    def __repr__(self) -> str:
        return "int | list[int]"


CLASS_ID = _ClassIdOrList()


def _is_valid_class_id(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, list):
        return len(value) > 0 and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
    return False


@dataclass(frozen=True)
class Tool:
    """One entry in the registry. `permitted_parameters` maps each allowed
    parameter name to its expected Python type -- the schema
    validate_parameters() enforces."""

    name: str
    description: str
    accepted_modalities: Modality
    gsd_range: tuple[float, float]  # (min_gsd_m, max_gsd_m), inclusive
    required_band_count: int
    permitted_parameters: dict[str, type | _ClassIdOrList]

    def __post_init__(self) -> None:
        if self.accepted_modalities not in ("optical", "sar", "both"):
            raise ValueError(f"invalid accepted_modalities: {self.accepted_modalities!r}")
        lo, hi = self.gsd_range
        if lo > hi:
            raise ValueError(f"gsd_range min {lo} exceeds max {hi}")
        if self.required_band_count < 1:
            raise ValueError("required_band_count must be >= 1")

    def accepts_modality(self, modality: str) -> bool:
        return self.accepted_modalities == "both" or self.accepted_modalities == modality

    def accepts_gsd(self, gsd_metres: float) -> bool:
        lo, hi = self.gsd_range
        return lo <= gsd_metres <= hi

    def accepts_band_count(self, band_count: int) -> bool:
        return band_count >= self.required_band_count


def _tool(
    name: str,
    description: str,
    accepted_modalities: Modality,
    gsd_range: tuple[float, float],
    required_band_count: int,
    permitted_parameters: dict[str, type | _ClassIdOrList],
) -> Tool:
    return Tool(name, description, accepted_modalities, gsd_range,
                required_band_count, permitted_parameters)


# --- Registration ------------------------------------------------------------
# segment, count, size, presence, adjacency, change: work on the class-ID
# raster (evidence/ops.py, evidence/change.py) so they tolerate either
# modality and only need the single-band class raster. caption and ground
# describe/locate what's visually in the scene, so they need real optical
# imagery with enough bands to be human-interpretable. cross_modal needs
# the model's full 16-channel input (not a pre-computed class raster) --
# it runs segmentation itself, three times, so it tolerates either
# modality but requires every real band the model expects.

REGISTRY: dict[str, Tool] = {
    tool.name: tool
    for tool in [
        _tool(
            "segment",
            "Semantic segmentation of raw imagery into the land-cover "
            "class-ID raster every other tool operates on.",
            "both", (1.0, 60.0), 1, {},
        ),
        _tool(
            "count",
            "Count connected components of a class (or union of classes) that clear a minimum area.",
            "both", (1.0, 60.0), 1, {"class_id": CLASS_ID, "min_area_m2": float},
        ),
        _tool(
            "size",
            "Total area covered by a class (or union of classes), in hectares.",
            "both", (1.0, 60.0), 1, {"class_id": CLASS_ID},
        ),
        _tool(
            "presence",
            "Whether a class (or union of classes) has any component clearing a minimum area.",
            "both", (1.0, 60.0), 1, {"class_id": CLASS_ID, "min_area_m2": float},
        ),
        _tool(
            "adjacency",
            "Whether class_a lies within a given distance of class_b (each a class or union of classes).",
            "both", (1.0, 60.0), 1, {"class_a": CLASS_ID, "class_b": CLASS_ID, "distance_m": float},
        ),
        _tool(
            "change",
            "Per-class area gained/lost and a change mask between two dates.",
            "both", (1.0, 60.0), 1, {},
        ),
        _tool(
            "caption",
            "Free-text description of the image, from a vision-language model.",
            "optical", (1.0, 30.0), 3, {"max_length": int},
        ),
        _tool(
            "ground",
            "Locate the image region matching a natural-language referring expression.",
            "optical", (1.0, 30.0), 3, {"query": str},
        ),
        _tool(
            "cross_modal",
            "Segments the same raw stack three ways (optical-only, SAR-only, "
            "fused) and reports which sensor(s) found each class.",
            "both", (1.0, 60.0), 16, {"cloud_simulation": bool},
        ),
        _tool(
            "verify",
            "Splits a generated answer into factual claims and re-tests each "
            "against the mask, per CLAUDE.md's 'No pixel, no claim'.",
            "both", (1.0, 60.0), 1, {"answer": str},
        ),
        _tool(
            "fusion",
            "Independently checks whether a class is present via four "
            "separate sources -- the segmentation head, the classification "
            "head, a real spectral index (NDWI/NDBI/NDVI), and the SAR "
            "branch where real SAR is available -- and reports per-source "
            "agreement plus a weighted fused verdict.",
            "both", (1.0, 60.0), 16, {"class_id": CLASS_ID},
        ),
        _tool(
            "metadata",
            "Answers a factual question about the scene's own record -- "
            "location, acquisition date, sensor/platform, resolution, or "
            "which classes are present -- read from the record, never "
            "asked of an LLM.",
            "both", (1.0, 60.0), 1, {"aspect": str},
        ),
        _tool(
            "conversational",
            "A short, scoped canned reply for a greeting or an off-topic "
            "question -- not a general chatbot, never asked of an LLM.",
            "both", (1.0, 60.0), 1, {"kind": str},
        ),
    ]
}


# --- Model/tool selection ----------------------------------------------------


@dataclass
class SelectionResult:
    tools: list[Tool]
    trace: list[TraceStep]


def select_model(image_metadata: dict) -> SelectionResult:
    """Route to the tools compatible with this image, by GSD, modality, and
    band count.

    image_metadata requires: gsd_metres (float), modality ("optical" or
    "sar"), band_count (int). Every registered tool is checked, and its
    accept/reject decision -- with the reason(s) for rejection -- is logged
    into the returned trace, so the routing itself is auditable.
    """
    gsd = image_metadata["gsd_metres"]
    modality = image_metadata["modality"]
    band_count = image_metadata["band_count"]

    selected: list[Tool] = []
    trace: list[TraceStep] = []

    for tool in REGISTRY.values():
        reasons = []
        if not tool.accepts_modality(modality):
            reasons.append(
                f"modality '{modality}' not accepted (tool accepts '{tool.accepted_modalities}')"
            )
        if not tool.accepts_gsd(gsd):
            reasons.append(f"gsd {gsd} m outside accepted range {tool.gsd_range}")
        if not tool.accepts_band_count(band_count):
            reasons.append(
                f"band_count {band_count} below required {tool.required_band_count}"
            )

        accepted = not reasons
        if accepted:
            selected.append(tool)

        trace.append(TraceStep(
            task=f"select_model:{tool.name}",
            tool="agent.registry.select_model",
            parameters={
                "candidate": tool.name,
                "gsd_metres": gsd,
                "modality": modality,
                "band_count": band_count,
            },
            output={"accepted": accepted, "reasons": reasons},
            confidence=1.0,
        ))

    return SelectionResult(tools=selected, trace=trace)


# --- Parameter validation -----------------------------------------------------


def validate_parameters(tool: Tool, parameters: dict[str, Any]) -> None:
    """Reject any parameter not in tool.permitted_parameters, by name, and
    any parameter whose value doesn't match the schema's declared type.
    Raises ParameterValidationError; returns None on success."""
    unknown = sorted(set(parameters) - set(tool.permitted_parameters))
    if unknown:
        raise ParameterValidationError(
            f"{tool.name!r}: unknown parameter(s) {unknown}; "
            f"permitted: {sorted(tool.permitted_parameters)}"
        )

    for param_name, value in parameters.items():
        expected_type = tool.permitted_parameters[param_name]

        if isinstance(expected_type, _ClassIdOrList):
            if not _is_valid_class_id(value):
                raise ParameterValidationError(
                    f"{tool.name!r}: parameter {param_name!r} expected an int or a "
                    f"non-empty list[int], got {value!r}"
                )
            continue

        # a whole-number float parameter is routinely written as a plain
        # int (500, not 500.0) in hand-written or LLM-emitted JSON -- that's
        # a valid float, so accept it. bool is excluded even though it's
        # technically an int subclass: True/False are never a valid float.
        if expected_type is float and isinstance(value, int) and not isinstance(value, bool):
            continue
        if not isinstance(value, expected_type):
            raise ParameterValidationError(
                f"{tool.name!r}: parameter {param_name!r} expected "
                f"{expected_type.__name__}, got {type(value).__name__}"
            )


# --- Capability table ---------------------------------------------------------
# Reference data: how big a real-world instance of each class typically is.
# This table is the only hardcoded thing here -- what's DERIVED (min
# resolvable size, and therefore countability) is always computed fresh from
# the image's own GSD, per CLAUDE.md: "Every area calculation reads GSD from
# image metadata. NEVER hardcode 10 m."

TYPICAL_OBJECT_SIZE_M: dict[str, float] = {
    "vehicle": 5.0,
    "building": 10.0,
    "water_body": 30.0,
    "agricultural_field": 150.0,
    "forest_patch": 200.0,
    # The real class names agent/vocabulary.py's NOUN_TO_CLASS resolves
    # built-up nouns to, once a caller passes the full 19-class
    # segmentation vocabulary instead of an illustrative 4-bucket scene
    # (e.g. "building" above) -- without these, the guardrail would
    # silently stop degrading building-like counts for any scene built
    # from agent.vocabulary.SEGMENTATION_CLASSES.
    "Urban fabric": 10.0,
    "Industrial or commercial units": 10.0,
}


@dataclass(frozen=True)
class ClassCapability:
    class_name: str
    typical_object_size_m: float
    min_resolvable_m: float
    countable: bool


@dataclass
class CapabilityTable:
    classes: dict[str, ClassCapability]
    trace: list[TraceStep]


def capability_table(image_metadata: dict) -> CapabilityTable:
    """Compute, from this image's GSD, which classes can honestly be
    counted.

    min_resolvable_m = 2.5 * gsd_metres (CLAUDE.md's minimum resolvable
    object size). A class is countable only if its typical real-world
    object size strictly exceeds that -- e.g. at 10 m GSD (min_resolvable
    = 25 m), buildings (10 m) are NOT countable but water bodies (30 m)
    are. This is computed fresh every call, never a hardcoded verdict, so
    it degrades correctly as GSD gets coarser (or improves as it gets
    finer).
    """
    gsd = image_metadata["gsd_metres"]
    min_resolvable_m = 2.5 * gsd

    classes: dict[str, ClassCapability] = {}
    trace: list[TraceStep] = []

    for class_name, typical_size in TYPICAL_OBJECT_SIZE_M.items():
        countable = typical_size > min_resolvable_m
        classes[class_name] = ClassCapability(
            class_name=class_name,
            typical_object_size_m=typical_size,
            min_resolvable_m=min_resolvable_m,
            countable=countable,
        )
        trace.append(TraceStep(
            task=f"capability_table:{class_name}",
            tool="agent.registry.capability_table",
            parameters={
                "class_name": class_name,
                "gsd_metres": gsd,
                "typical_object_size_m": typical_size,
            },
            output={"min_resolvable_m": min_resolvable_m, "countable": countable},
            confidence=1.0,
        ))

    return CapabilityTable(classes=classes, trace=trace)
