"""SatQuery AI's HTTP API: FastAPI over the same classify_task -> planner
-> validate -> executor pipeline every script/eval in this repo uses --
this module adds no new reasoning, it's wiring: HTTP in, the real
pipeline, evidence.schema.Evidence out.

Endpoints
-----------
POST /upload  -- validate an uploaded image (or pair) via
                 raster_io/validate.py, then run the FULL pipeline on it:
                 perception/infer.py, then register a new scene so the
                 returned `scene_id` is immediately queryable via /query,
                 exactly like a preloaded demo scene. Three shapes:
                   - one file: a "single" scene, sensor_id="optical".
                     Fewer than 16 real channels (true of almost every
                     upload) is handled honestly, not hidden -- see
                     api/scenes.py's build_model_input_from_bands.
                   - two files + pair_kind="cross_modal": one 'optical'
                     and one 'sar' image, phase-correlation co-registered,
                     combined into one real dual-sensor stack
                     (sensor_id="fused").
                   - two files + pair_kind="change": a bi-temporal pair,
                     each segmented independently; the scene's primary
                     mask is the AFTER date, with the BEFORE date kept
                     alongside for the 'change' tool.
                 A pair that fails co-registration by more than the
                 threshold does NOT hard-reject (per raster_io/validate.py's
                 own design) -- it registers the first image alone as a
                 'single' scene instead, with a warning explaining why.
POST /query   -- answer a natural-language question against a scene:
                 classify_task -> plan_from_query -> validate ->
                 executor.run(), wrapped as an evidence.schema.Evidence
                 plus any capability-guardrail limitation. Every /query is
                 logged to SQLite so its report can be re-downloaded later.
GET  /scenes  -- lists every scene -- preloaded demos AND uploads --
                 including each one's `warnings` (e.g. missing channels).
GET  /report  -- re-serves a past /query's full response as a
                 downloadable .json file, by the report_id /query returned.

GET  /scenes/{id}/image  -- the scene's true-color preview, as a PNG
                 rendered from its real bands (contrast-stretched for
                 display only).
GET  /scenes/{id}/mask   -- the scene's predicted class mask, colorized,
                 as a PNG -- same palette scripts/demo_real.py uses. For a
                 'change'-kind scene this is the AFTER (current) mask.
GET  /scenes/{id}/legend -- {class_id, class_name, color} for every class
                 actually present in that scene's mask (not all 19).

GET  /scenes/{id}/cross-modal      -- the full tools/cross_modal.py
                 comparison (per-class optical/SAR/fused attribution, plus
                 a USABLE/INSUFFICIENT verdict per sensor) for a
                 'cross_modal'-kind scene, computed on request from its raw
                 stack. `cloud_simulation`/`cloud_fraction` query params
                 replay the same demo-mode degradation tools/cross_modal.py
                 exposes.
GET  /scenes/{id}/cross-modal-mask -- one sensor's (`sensor=optical|sar|
                 fused`) colorized predicted-mask PNG, for the CROSS-MODAL
                 COMPARISON view's swipe divider. Same cloud_simulation
                 params as above.

Per CLAUDE.md: SQLite (not PostgreSQL) for scene/query metadata, plain
.npy files on disk for masks/stacks, no Celery -- inference runs
synchronously, in-request for an upload, and once per demo patch at
startup. classify_task/plan_from_query are NOT forced offline here
(unlike the demo scripts) -- this is the real product path, so Gemini is
used when GOOGLE_API_KEY is configured, falling back to the deterministic
keyword parser exactly as agent/planner.py already does when it isn't.

Run with the venv's own interpreter:
    venv\\Scripts\\python.exe -m uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs.
"""

import json
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from agent.dsl import PlanValidationError, validate
from agent.executor import ExecutorError, run as run_plan
from agent.planner import PlannerError, SceneDescriptor, plan_from_query
from agent.tasks import classify_task
from api.database import get_connection
from api.rendering import build_answer, build_confidence, build_geometry
from api.schemas import (
    ChangeSummary,
    ClassAreaChange,
    CrossModalFindingOut,
    CrossModalSummary,
    QueryRequest,
    QueryResponse,
    SceneSummary,
    UploadResponse,
)
from api.scenes import (
    class_color,
    ensure_demo_change_scene_loaded,
    ensure_demo_cross_modal_scene_loaded,
    ensure_scenes_loaded,
    load_scene_before_stack_for_row,
    load_scene_mask,
    load_scene_stack_for_row,
    register_change_pair,
    register_cross_modal_pair,
    register_single_scene,
    render_change_mask_png,
    render_mask_png,
    render_preview_png,
)
from evidence.change import area_changes, change_mask, summarize as summarize_change
from evidence.schema import Evidence
from perception import infer
from raster_io.readers import RasterImage, ReaderError, read_image
from raster_io.validate import ImageInput, validate_images
from tools import cross_modal as cross_modal_tool

ROOT_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = ROOT_DIR / "data" / "api" / "uploads"


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_scenes_loaded()
    ensure_demo_change_scene_loaded()
    ensure_demo_cross_modal_scene_loaded()
    yield


app = FastAPI(
    title="SatQuery AI",
    description="Agentic vision-language assistant for satellite imagery. No pixel, no claim.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # wildcard origins + credentials is a browser-rejected combination
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_scene_row(scene_id: str):
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM scenes WHERE id = ?", (scene_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail=f"unknown scene_id {scene_id!r}")
    return row


# Loading the ONNX session/class config isn't cheap -- cache it across
# requests the same way api/scenes.py's ensure_scenes_loaded() does,
# rather than reloading it for every single upload.
_inference_session: Any = None
_inference_config: Any = None


def _get_inference_session() -> tuple[Any, Any]:
    global _inference_session, _inference_config
    if _inference_session is None:
        _inference_session = infer.load_model()
        _inference_config = infer.load_class_config()
    return _inference_session, _inference_config


def _save_upload(file: UploadFile) -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS_DIR / f"{uuid.uuid4().hex[:8]}_{file.filename}"
    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return dest


def _read_upload(file: UploadFile, gsd_metres: float | None) -> tuple[RasterImage | None, str | None]:
    """Saves and reads one uploaded file. Returns (image, None) on
    success or (None, reason) on failure -- never raises, since a bad
    upload is an expected, valid outcome, not a server error."""
    dest = _save_upload(file)
    try:
        return read_image(dest, gsd_metres_override=gsd_metres), None
    except ReaderError as exc:
        return None, str(exc)


@app.post("/upload", response_model=UploadResponse)
async def upload_image(
    file: UploadFile = File(..., description="The image (single upload, or the first of a pair)"),
    modality: str = Form("optical", description="'optical' or 'sar'"),
    gsd_metres: float | None = Form(None, description="Required for PNG/JPEG, which carry no georeferencing"),
    file2: UploadFile | None = File(None, description="Second image, for a cross_modal or change pair"),
    modality2: str = Form("sar", description="file2's modality -- 'optical' or 'sar'"),
    gsd_metres2: float | None = Form(None),
    pair_kind: str | None = Form(
        None, description="Required when file2 is given: 'cross_modal' (optical+SAR) or 'change' (bi-temporal)"
    ),
    before_date: str | None = Form(
        None, description="Free-text acquisition date for file (the BEFORE image), a 'change' pair only"
    ),
    after_date: str | None = Form(
        None, description="Free-text acquisition date for file2 (the AFTER image), a 'change' pair only"
    ),
) -> UploadResponse:
    """Validate (raster_io/validate.py) then run the full pipeline: infer,
    register a new scene, return its `scene_id` ready to /query."""
    image, error = _read_upload(file, gsd_metres)
    if error:
        return UploadResponse(ok=False, reason=error, modality=modality)

    if file2 is None:
        return _handle_single_upload(image, modality)

    if pair_kind not in ("cross_modal", "change"):
        return UploadResponse(
            ok=False, modality=modality,
            reason=f"pair_kind must be 'cross_modal' or 'change' when a second file is given, got {pair_kind!r}",
        )

    image2, error2 = _read_upload(file2, gsd_metres2)
    if error2:
        return UploadResponse(ok=False, reason=error2, modality=modality)

    if pair_kind == "cross_modal":
        return _handle_cross_modal_upload(image, modality, image2, modality2)
    return _handle_change_upload(image, modality, image2, modality2, before_date, after_date)


def _handle_single_upload(image: RasterImage, modality: str) -> UploadResponse:
    result = validate_images([ImageInput(image, modality)], expected_count=1)
    if not result.ok:
        return UploadResponse(
            ok=False, reason=result.reason, modality=modality, gsd_metres=image.gsd_metres,
            band_count=image.band_count, width=image.width, height=image.height,
        )

    session, config = _get_inference_session()
    scene_id, warnings = register_single_scene(image, session, config)
    return UploadResponse(
        ok=True, modality=modality, gsd_metres=image.gsd_metres, band_count=image.band_count,
        width=image.width, height=image.height, scene_id=scene_id, kind="single", warnings=warnings,
    )


def _handle_cross_modal_upload(
    image: RasterImage, modality: str, image2: RasterImage, modality2: str,
) -> UploadResponse:
    declared = {modality, modality2}
    if declared != {"optical", "sar"}:
        return UploadResponse(
            ok=False, modality=modality,
            reason=f"a cross_modal pair needs one 'optical' and one 'sar' image, got {sorted(declared)}",
        )

    result = validate_images(
        [ImageInput(image, modality), ImageInput(image2, modality2)],
        expected_count=2, require_matching_crs=False,
    )
    if not result.ok:
        return UploadResponse(ok=False, reason=result.reason, modality=modality)

    session, config = _get_inference_session()

    if result.fallback_single_modality:
        # Too misaligned to trust as a pair -- per raster_io/validate.py's
        # own design this is not a hard rejection: fall back to
        # registering the optical (reference) image alone, disclosed.
        optical_image = image if modality == "optical" else image2
        scene_id, warnings = register_single_scene(optical_image, session, config)
        return UploadResponse(
            ok=True, modality=modality, scene_id=scene_id, kind="single",
            warnings=[result.reason] + warnings, fallback_single_modality=True,
        )

    images_by_modality = {modality: image, modality2: image2}
    if result.shifted_images is not None:
        images_by_modality = {modality: result.shifted_images[0], modality2: result.shifted_images[1]}

    scene_id, warnings = register_cross_modal_pair(
        images_by_modality["optical"], images_by_modality["sar"], session, config,
    )
    return UploadResponse(
        ok=True, modality=modality, scene_id=scene_id, kind="cross_modal",
        warnings=warnings, auto_shifted=result.auto_shifted,
    )


def _handle_change_upload(
    image: RasterImage, modality: str, image2: RasterImage, modality2: str,
    before_date: str | None = None, after_date: str | None = None,
) -> UploadResponse:
    result = validate_images(
        [ImageInput(image, modality), ImageInput(image2, modality2)],
        expected_count=2, require_matching_crs=False,
    )
    if not result.ok:
        return UploadResponse(ok=False, reason=result.reason, modality=modality)

    session, config = _get_inference_session()

    if result.fallback_single_modality:
        scene_id, warnings = register_single_scene(image, session, config)
        return UploadResponse(
            ok=True, modality=modality, scene_id=scene_id, kind="single",
            warnings=[result.reason] + warnings, fallback_single_modality=True,
        )

    before_image, after_image = (image, image2) if result.shifted_images is None else result.shifted_images
    scene_id, warnings = register_change_pair(
        before_image, after_image, session, config, before_date=before_date, after_date=after_date,
    )
    return UploadResponse(
        ok=True, modality=modality, scene_id=scene_id, kind="change",
        warnings=warnings, auto_shifted=result.auto_shifted,
    )


@app.get("/scenes", response_model=list[SceneSummary])
def list_scenes() -> list[SceneSummary]:
    """Every scene -- preloaded demos and uploads alike -- ready to
    /query against by its `id`."""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM scenes ORDER BY created_at").fetchall()
    finally:
        conn.close()
    return [
        SceneSummary(
            id=row["id"], filename=row["filename"], width=row["width"], height=row["height"],
            gsd_metres=row["gsd_metres"], sensor=row["sensor"], kind=row["kind"], source=row["source"],
            warnings=json.loads(row["warnings_json"]),
            before_date=row["before_date"], after_date=row["after_date"],
        )
        for row in rows
    ]


@app.get("/scenes/{scene_id}/image")
def scene_image(
    scene_id: str,
    when: Literal["before", "after"] = "after",
    cloud_simulation: bool = False,
    cloud_fraction: float = cross_modal_tool.DEFAULT_CLOUD_FRACTION,
) -> Response:
    """The scene's true-color preview PNG, rendered on request from its
    real bands (see api/scenes.py's render_preview_png). For an uploaded
    scene this is rendered from the exact (possibly channel-gapped) stack
    the model actually saw -- what you see is what it saw. `when="before"`
    is only meaningful for a 'change'-kind scene (the CHANGE COMPARISON
    view's other side); every other scene has only one date and 404s for it.
    `cloud_simulation=true` renders the same simulated-cloud-cover stack
    (tools/cross_modal.py's simulate_cloud_cover) the CROSS-MODAL COMPARISON
    view's demo-mode toggle segments -- so the preview visibly shows the
    zeroed optical rows, not just the resulting mask."""
    row = _get_scene_row(scene_id)
    stack = load_scene_before_stack_for_row(row) if when == "before" else load_scene_stack_for_row(row)
    if stack is None:
        raise HTTPException(status_code=404, detail=f"no {when} stack available for scene {scene_id!r}")
    if cloud_simulation:
        stack = cross_modal_tool.simulate_cloud_cover(stack, cloud_fraction)
    return Response(content=render_preview_png(stack), media_type="image/png")


@app.get("/scenes/{scene_id}/mask")
def scene_mask_image(scene_id: str, when: Literal["before", "after"] = "after") -> Response:
    """The scene's predicted class mask, colorized, as a PNG. For a
    'change'-kind scene, `when="after"` (the default) is the current mask
    and `when="before"` is the earlier date; every other scene has only one
    date and 404s for `when="before"`."""
    row = _get_scene_row(scene_id)
    if when == "before":
        if not row["before_mask_path"]:
            raise HTTPException(status_code=404, detail=f"no before mask available for scene {scene_id!r}")
        mask = load_scene_mask(row["before_mask_path"])
    else:
        mask = load_scene_mask(row["mask_path"])
    return Response(content=render_mask_png(mask), media_type="image/png")


@app.get("/scenes/{scene_id}/change-mask")
def scene_change_mask_image(scene_id: str) -> Response:
    """A transparent-except-changed-pixels RGBA PNG (api/scenes.py's
    render_change_mask_png), for the CHANGE COMPARISON view's toggleable
    overlay. 404 for any scene that isn't a 'change' pair."""
    row = _get_scene_row(scene_id)
    if row["kind"] != "change" or not row["before_mask_path"]:
        raise HTTPException(status_code=404, detail=f"scene {scene_id!r} is not a change pair")
    before = load_scene_mask(row["before_mask_path"])
    after = load_scene_mask(row["mask_path"])
    return Response(content=render_change_mask_png(change_mask(before, after)), media_type="image/png")


@app.get("/scenes/{scene_id}/change", response_model=ChangeSummary)
def scene_change_summary(scene_id: str) -> ChangeSummary:
    """The full per-class gain/loss breakdown (evidence/change.py's
    area_changes, in hectares) for a 'change'-kind scene, plus the overall
    changed-pixel count/fraction and a plain-English summary -- independent
    of /query, so the CHANGE COMPARISON view's table is always populated
    regardless of which (if any) question was asked. 404 for any scene
    that isn't a 'change' pair."""
    row = _get_scene_row(scene_id)
    if row["kind"] != "change" or not row["before_mask_path"]:
        raise HTTPException(status_code=404, detail=f"scene {scene_id!r} is not a change pair")

    before = load_scene_mask(row["before_mask_path"])
    after = load_scene_mask(row["mask_path"])
    classes = {int(k): v for k, v in json.loads(row["classes_json"]).items()}
    metadata = {"gsd_metres": row["gsd_metres"]}

    deltas = area_changes(before, after, metadata)
    changed = change_mask(before, after)
    classes_out = [
        ClassAreaChange(
            class_id=class_id, class_name=classes.get(class_id, str(class_id)),
            gained_ha=d["gained_ha"], lost_ha=d["lost_ha"], net_ha=d["gained_ha"] - d["lost_ha"],
        )
        for class_id, d in deltas.items()
        if d["gained_ha"] > 0 or d["lost_ha"] > 0
    ]
    classes_out.sort(key=lambda c: abs(c.net_ha), reverse=True)
    summary = summarize_change(before, after, metadata, classes)

    return ChangeSummary(
        summary=summary, changed_pixels=int(changed.sum()),
        changed_fraction=float(changed.mean()), classes=classes_out,
    )


@app.get("/scenes/{scene_id}/legend")
def scene_legend(scene_id: str) -> list[dict]:
    """{class_id, class_name, color} for every class actually present in
    this scene's mask, sorted by area descending."""
    row = _get_scene_row(scene_id)
    mask = load_scene_mask(row["mask_path"])
    classes = {int(k): v for k, v in json.loads(row["classes_json"]).items()}

    present_ids, counts = np.unique(mask, return_counts=True)
    order = np.argsort(counts)[::-1]
    return [
        {
            "class_id": int(present_ids[i]),
            "class_name": classes.get(int(present_ids[i]), str(present_ids[i])),
            "color": "#%02x%02x%02x" % class_color(int(present_ids[i])),
        }
        for i in order
    ]


def _get_cross_modal_stack(scene_id: str) -> tuple[Any, dict]:
    """Shared 404/stack-lookup for both cross-modal endpoints below: the
    scene must be kind='cross_modal' and have a raw 16-channel stack to
    segment (tools/cross_modal.py needs the full model input, not a
    pre-computed mask -- see agent/executor.py's own cross_modal wiring)."""
    row = _get_scene_row(scene_id)
    if row["kind"] != "cross_modal":
        raise HTTPException(status_code=404, detail=f"scene {scene_id!r} is not a cross_modal scene")
    stack = load_scene_stack_for_row(row)
    if stack is None:
        raise HTTPException(status_code=404, detail=f"no stack available for scene {scene_id!r}")
    return stack, {"gsd_metres": row["gsd_metres"]}


@app.get("/scenes/{scene_id}/cross-modal", response_model=CrossModalSummary)
def scene_cross_modal(
    scene_id: str,
    cloud_simulation: bool = False,
    cloud_fraction: float = cross_modal_tool.DEFAULT_CLOUD_FRACTION,
) -> CrossModalSummary:
    """tools/cross_modal.py's full comparison -- every real area number
    (evidence/ops.py's size(), over each of the three real predicted masks)
    plus the per-sensor USABLE/INSUFFICIENT verdict -- for the CROSS-MODAL
    COMPARISON view's findings table. 404 for any scene that isn't a
    'cross_modal' pair."""
    stack, metadata = _get_cross_modal_stack(scene_id)
    session, config = _get_inference_session()
    result = cross_modal_tool.cross_modal_analysis(
        stack, metadata, cloud_simulation=cloud_simulation, cloud_fraction=cloud_fraction,
        session=session, config=config,
    )
    return CrossModalSummary(
        summary=result.summary,
        cloud_simulated=result.cloud_simulated,
        cloud_fraction=result.cloud_fraction,
        sensor_status=result.sensor_status,
        findings=[
            CrossModalFindingOut(
                class_id=f.class_id, class_name=f.class_name,
                optical_area_ha=f.optical_area_ha, sar_area_ha=f.sar_area_ha, fused_area_ha=f.fused_area_ha,
                detected_by=list(f.detected_by), attribution=f.attribution,
            )
            for f in result.findings
        ],
    )


@app.get("/scenes/{scene_id}/cross-modal-mask")
def scene_cross_modal_mask(
    scene_id: str,
    sensor: Literal["optical", "sar", "fused"] = "fused",
    cloud_simulation: bool = False,
    cloud_fraction: float = cross_modal_tool.DEFAULT_CLOUD_FRACTION,
) -> Response:
    """The colorized predicted-mask PNG for one sensor path (optical-only,
    SAR-only, or fused) over this scene's raw stack -- the CROSS-MODAL
    COMPARISON view's swipe divider clips this against the true-color
    preview from GET /scenes/{id}/image. `cloud_simulation=true` segments
    the same simulated-cloud-cover stack that endpoint can also render, so
    the two stay comparable. 404 for any scene that isn't a 'cross_modal'
    pair."""
    stack, _metadata = _get_cross_modal_stack(scene_id)
    if cloud_simulation:
        stack = cross_modal_tool.simulate_cloud_cover(stack, cloud_fraction)
    session, config = _get_inference_session()
    result = infer.segment_image(stack, sensor, session=session, config=config)
    return Response(content=render_mask_png(result.mask), media_type="image/png")


@app.post("/query", response_model=QueryResponse)
def query_scene(request: QueryRequest) -> QueryResponse:
    """classify_task -> plan_from_query -> validate -> executor.run()
    against `request.scene_id`, returning an evidence.schema.Evidence
    (answer, geometry, confidence, execution trace) plus any capability-
    guardrail limitation. Every raster a scene actually has (mask; before/
    after, for a 'change' scene; the raw stack, for 'cross_modal'
    questions) is handed to the executor -- it uses whichever the emitted
    plan's tool actually needs. Logged to SQLite so GET /report can
    re-serve it later by the returned `report_id`."""
    row = _get_scene_row(request.scene_id)
    mask = load_scene_mask(row["mask_path"])
    before = load_scene_mask(row["before_mask_path"]) if row["before_mask_path"] else None
    after = mask if before is not None else None
    stack = load_scene_stack_for_row(row)
    classes = {int(k): v for k, v in json.loads(row["classes_json"]).items()}
    # filename/sensor enrich the 'metadata' tool's own scene-record lookup
    # (evidence/metadata.py) -- every other tool ignores these extra keys.
    metadata = {"gsd_metres": row["gsd_metres"], "filename": row["filename"], "sensor": row["sensor"]}

    scene = SceneDescriptor(
        layers=["class_raster"], classes=classes, sensor=row["sensor"],
        gsd_metres=row["gsd_metres"], bbox=(0.0, 0.0, 0.0, 0.0),  # these scenes carry no geotransform
    )

    classification = classify_task(request.query)
    try:
        plan = plan_from_query(request.query, scene)
        validate(plan)
        _, exec_trace = run_plan(plan, metadata, mask=mask, before=before, after=after, stack=stack, classes=classes)
    except (PlannerError, PlanValidationError, ExecutorError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    full_trace = list(classification.trace) + list(exec_trace)
    value, units, _answer_text, limitation = build_answer(full_trace, classes)
    final_step = next(step for step in reversed(full_trace) if step.task.startswith("execute:"))
    geometry = build_geometry(mask, classes, final_step)
    confidence = build_confidence(classification, full_trace)

    evidence = Evidence(value=value, units=units, geometry=geometry, confidence=confidence,
                         execution_trace=full_trace)
    response = QueryResponse(
        report_id=str(uuid.uuid4()), scene_id=request.scene_id, query=request.query,
        evidence=evidence, limitation=limitation,
    )

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO query_log (id, scene_id, query, created_at, response_json) VALUES (?, ?, ?, ?, ?)",
            (response.report_id, request.scene_id, request.query,
             datetime.now(timezone.utc).isoformat(), response.model_dump_json()),
        )
        conn.commit()
    finally:
        conn.close()

    return response


@app.get("/report")
def download_report(report_id: str) -> Response:
    """Re-serve a past /query response, byte-identical, as a downloadable
    .json file -- no recomputation, just what was actually logged."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT response_json FROM query_log WHERE id = ?", (report_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail=f"unknown report_id {report_id!r}")

    return Response(
        content=row["response_json"],
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="report_{report_id}.json"'},
    )
