"""Pydantic request/response models for api/main.py's endpoints.

/query and /report wrap evidence.schema.Evidence -- the same typed
contract every other consumer of SatQuery AI's answers depends on --
rather than inventing a parallel response shape.
"""

from pydantic import BaseModel

from evidence.schema import Evidence


class SceneSummary(BaseModel):
    id: str
    filename: str
    width: int
    height: int
    gsd_metres: float
    sensor: str


class QueryRequest(BaseModel):
    scene_id: str
    query: str


class QueryResponse(BaseModel):
    report_id: str
    scene_id: str
    query: str
    evidence: Evidence
    limitation: str | None = None


class UploadResponse(BaseModel):
    ok: bool
    reason: str | None = None
    modality: str
    gsd_metres: float | None = None
    band_count: int | None = None
    width: int | None = None
    height: int | None = None
    auto_shifted: bool = False
    fallback_single_modality: bool = False
