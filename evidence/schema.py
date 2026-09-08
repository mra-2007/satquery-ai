"""The Evidence JSON schema.

This is a contract: the API layer serializes it, the frontend renders it, and
every downstream consumer of SatQuery AI's answers depends on this exact
shape. Changing a field name or type here is a breaking change for all of
them. Extend by adding new optional fields; do not rename or repurpose
existing ones. `Evidence.schema_version` exists precisely so a breaking
change, if it's ever unavoidable, is at least detectable by consumers.

Per CLAUDE.md's "No pixel, no claim.": the LLM never invents `value`. It is
always produced by deterministic geometry (evidence/ops.py, evidence/change.py)
over a real raster, `geometry` is the actual GeoJSON footprint that geometry
ran over, and `execution_trace` is the literal sequence of tool calls that
produced it -- task, tool, parameters, output, confidence, exactly as
mandated by CLAUDE.md's execution-trace requirement.

    from evidence.schema import Evidence, TraceStep, SourceConfidence

    Evidence(
        value=3,
        units="ponds",
        geometry={
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]},
                "properties": {"class": "water"},
            }],
        },
        confidence=[SourceConfidence(source="onnx-landcover-classifier", confidence=0.94)],
        execution_trace=[
            TraceStep(
                task="count water bodies",
                tool="evidence.ops.count",
                parameters={"class_id": 1, "min_area_m2": 500},
                output=3,
                confidence=0.94,
            )
        ],
    )
"""

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

# --- GeoJSON (RFC 7946), just the subset SatQuery AI actually emits ---------

Position = Union[tuple[float, float], tuple[float, float, float]]


class _GeoJSONModel(BaseModel):
    """Base for every GeoJSON object: reject unknown fields so a typo in a
    tool's output fails loudly instead of silently dropping geometry."""

    model_config = ConfigDict(extra="forbid")


class Point(_GeoJSONModel):
    type: Literal["Point"] = "Point"
    coordinates: Position


class MultiPoint(_GeoJSONModel):
    type: Literal["MultiPoint"] = "MultiPoint"
    coordinates: list[Position]


class LineString(_GeoJSONModel):
    type: Literal["LineString"] = "LineString"
    coordinates: list[Position]


class MultiLineString(_GeoJSONModel):
    type: Literal["MultiLineString"] = "MultiLineString"
    coordinates: list[list[Position]]


class Polygon(_GeoJSONModel):
    """`coordinates` is a list of linear rings: the first is the exterior
    ring, any further rings are holes, each ring closed (first point ==
    last point), per RFC 7946."""

    type: Literal["Polygon"] = "Polygon"
    coordinates: list[list[Position]]


class MultiPolygon(_GeoJSONModel):
    type: Literal["MultiPolygon"] = "MultiPolygon"
    coordinates: list[list[list[Position]]]


Geometry = Annotated[
    Union[Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon],
    Field(discriminator="type"),
]


class GeometryCollection(_GeoJSONModel):
    type: Literal["GeometryCollection"] = "GeometryCollection"
    geometries: list[Geometry]


class Feature(_GeoJSONModel):
    type: Literal["Feature"] = "Feature"
    geometry: Union[Geometry, GeometryCollection, None] = None
    properties: dict[str, Any] | None = None
    id: str | int | None = None


class FeatureCollection(_GeoJSONModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[Feature]


GeoJSON = Annotated[
    Union[Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon,
          GeometryCollection, Feature, FeatureCollection],
    Field(discriminator="type"),
]


# --- Confidence and execution trace -----------------------------------------


class SourceConfidence(BaseModel):
    """Confidence contributed by one source (a model, tool, or heuristic).
    `value` is typically backed by several sources -- e.g. a land-cover
    classifier's confidence AND a geometry op's own certainty -- so
    `Evidence.confidence` is a list, one entry per contributing source."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(
        ..., description="Name of the model/tool/heuristic, e.g. "
        "'onnx-landcover-classifier' or 'evidence.ops.count'."
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in [0, 1].")


class TraceStep(BaseModel):
    """One entry in the execution trace. CLAUDE.md mandates that every tool
    call be recorded as exactly these five fields -- this model IS that
    mandate, enforced in code."""

    model_config = ConfigDict(extra="forbid")

    task: str = Field(..., description="The sub-task this tool call was performed for.")
    tool: str = Field(..., description="Fully-qualified name of the tool that was called, "
                                        "e.g. 'evidence.ops.count'.")
    parameters: dict[str, Any] = Field(..., description="Keyword arguments the tool was called with.")
    output: Any = Field(..., description="The tool's raw return value.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this step's output, in [0, 1].")


# --- Evidence: the top-level contract ---------------------------------------


class Evidence(BaseModel):
    """The Evidence JSON returned for every answer SatQuery AI gives.

    Fields:
      value:            The answer itself. Always produced by deterministic
                         geometry, never by the LLM.
      units:             Units `value` is expressed in (e.g. "hectares",
                         "ponds", "m"), or None for a plain count/boolean.
      geometry:          The GeoJSON that `value` was computed over -- the
                         actual pixels/polygons responsible for the answer,
                         so the answer can be inspected and never taken on
                         faith.
      confidence:        Per-source confidence contributing to `value`.
      execution_trace:   The literal, ordered sequence of tool calls that
                         produced `value`.
      schema_version:    Contract version. Bump only on a breaking change.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    value: bool | int | float | str = Field(..., description="The answer value.")
    units: str | None = Field(None, description="Units of `value`, or None if unitless.")
    geometry: GeoJSON = Field(..., description="Contributing geometry, as GeoJSON.")
    confidence: list[SourceConfidence] = Field(..., description="Per-source confidence.")
    execution_trace: list[TraceStep] = Field(..., description="Ordered tool-call trace.")
