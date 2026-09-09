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
    kind: str = "single"  # "single" | "cross_modal" | "change"
    source: str = "demo"  # "demo" | "upload"
    warnings: list[str] = []
    # Free-text, caller-supplied acquisition dates for a 'change'-kind
    # scene -- None whenever not supplied (never fabricated; see
    # api/scenes.py's register_change_pair docstring).
    before_date: str | None = None
    after_date: str | None = None


class ClassAreaChange(BaseModel):
    """One class's gained/lost area (hectares) between the before and
    after dates of a 'change'-kind scene -- straight from
    evidence.change.area_changes, nothing computed here."""

    class_id: int
    class_name: str
    gained_ha: float
    lost_ha: float
    net_ha: float


class ChangeSummary(BaseModel):
    """The full per-class gain/loss breakdown for a 'change'-kind scene,
    for the CHANGE COMPARISON view's table -- independent of /query, so
    it's always available regardless of which question (if any) was asked."""

    summary: str
    changed_pixels: int
    changed_fraction: float
    classes: list[ClassAreaChange]


class CrossModalFindingOut(BaseModel):
    """One class's per-sensor attribution, straight from
    tools/cross_modal.py's CrossModalFinding -- nothing computed here."""

    class_id: int
    class_name: str
    optical_area_ha: float
    sar_area_ha: float
    fused_area_ha: float
    detected_by: list[str]
    attribution: str


class CrossModalSummary(BaseModel):
    """The full OPTICAL/SAR/FUSED comparison for a 'cross_modal'-kind
    scene, for the CROSS-MODAL COMPARISON view -- independent of /query, so
    its three toggles and findings table are always populated regardless of
    which (if any) question was asked."""

    summary: str
    cloud_simulated: bool
    cloud_fraction: float | None
    sensor_status: dict[str, str]
    findings: list[CrossModalFindingOut]


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
    # Populated only once validation passed AND the scene was registered --
    # i.e. exactly when the frontend can /query this upload immediately.
    scene_id: str | None = None
    kind: str | None = None  # "single" | "cross_modal" | "change"
    warnings: list[str] = []
