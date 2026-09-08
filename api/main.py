"""SatQuery AI's HTTP API: FastAPI over the same classify_task -> planner
-> validate -> executor pipeline every script/eval in this repo uses --
this module adds no new reasoning, it's wiring: HTTP in, the real
pipeline, evidence.schema.Evidence out.

Endpoints
-----------
POST /upload  -- validate an uploaded image via raster_io/validate.py.
                 Does NOT turn the upload into a queryable scene (that
                 would need building a full 16-channel model input from
                 whatever bands were uploaded, the way eval/run_rsvqa.py
                 does for RGB-only sources) -- this endpoint's job is
                 exactly what CLAUDE.md's pre-flight checks are for:
                 confirm the image is usable before anything downstream
                 ever touches it.
POST /query   -- answer a natural-language question against a preloaded
                 scene's mask: classify_task -> plan_from_query ->
                 validate -> executor.run(), wrapped as an
                 evidence.schema.Evidence plus any capability-guardrail
                 limitation. Every /query is logged to SQLite so its
                 report can be re-downloaded later.
GET  /scenes  -- lists the demo scenes loaded from data/demo_patches/*.npz
                 (api/scenes.py runs inference on them once, at startup).
GET  /report  -- re-serves a past /query's full response as a
                 downloadable .json file, by the report_id /query returned.

Per CLAUDE.md: SQLite (not PostgreSQL) for scene/query metadata, plain
.npy files on disk for masks, no Celery -- scene inference runs
synchronously at startup, not as a background job. classify_task/
plan_from_query are NOT forced offline here (unlike the demo scripts) --
this is the real product path, so Gemini is used when GOOGLE_API_KEY is
configured, falling back to the deterministic keyword parser exactly as
agent/planner.py already does when it isn't.

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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from agent.dsl import PlanValidationError, validate
from agent.executor import ExecutorError, run as run_plan
from agent.planner import PlannerError, SceneDescriptor, plan_from_query
from agent.tasks import classify_task
from api.database import get_connection
from api.rendering import build_answer, build_confidence, build_geometry
from api.schemas import QueryRequest, QueryResponse, SceneSummary, UploadResponse
from api.scenes import ensure_scenes_loaded, load_scene_mask
from evidence.schema import Evidence
from raster_io.readers import ReaderError, read_image
from raster_io.validate import ImageInput, validate_images

ROOT_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = ROOT_DIR / "data" / "api" / "uploads"


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_scenes_loaded()
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


@app.post("/upload", response_model=UploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    modality: str = Form("optical", description="'optical' or 'sar'"),
    gsd_metres: float | None = Form(None, description="Required for PNG/JPEG, which carry no georeferencing"),
) -> UploadResponse:
    """Save the upload, read it, and run it through
    raster_io/validate.py's pre-flight checks. Returns ok=False with a
    reason on any failure -- never raises for a bad image, since a
    rejected upload is an expected, valid outcome, not a server error."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS_DIR / file.filename
    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        image = read_image(dest, gsd_metres_override=gsd_metres)
    except ReaderError as exc:
        return UploadResponse(ok=False, reason=str(exc), modality=modality)

    result = validate_images([ImageInput(image, modality)], expected_count=1)
    return UploadResponse(
        ok=result.ok,
        reason=result.reason,
        modality=modality,
        gsd_metres=image.gsd_metres,
        band_count=image.band_count,
        width=image.width,
        height=image.height,
        auto_shifted=result.auto_shifted,
        fallback_single_modality=result.fallback_single_modality,
    )


@app.get("/scenes", response_model=list[SceneSummary])
def list_scenes() -> list[SceneSummary]:
    """The preloaded demo scenes (data/demo_patches/*.npz), ready to
    /query against by their `id`."""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM scenes ORDER BY id").fetchall()
    finally:
        conn.close()
    return [
        SceneSummary(
            id=row["id"], filename=row["filename"], width=row["width"], height=row["height"],
            gsd_metres=row["gsd_metres"], sensor=row["sensor"],
        )
        for row in rows
    ]


@app.post("/query", response_model=QueryResponse)
def query_scene(request: QueryRequest) -> QueryResponse:
    """classify_task -> plan_from_query -> validate -> executor.run()
    against `request.scene_id`'s preloaded mask, returning an
    evidence.schema.Evidence (answer, geometry, confidence, execution
    trace) plus any capability-guardrail limitation. Logged to SQLite so
    GET /report can re-serve it later by the returned `report_id`."""
    row = _get_scene_row(request.scene_id)
    mask = load_scene_mask(row["mask_path"])
    classes = {int(k): v for k, v in json.loads(row["classes_json"]).items()}
    metadata = {"gsd_metres": row["gsd_metres"]}

    scene = SceneDescriptor(
        layers=["class_raster"], classes=classes, sensor=row["sensor"],
        gsd_metres=row["gsd_metres"], bbox=(0.0, 0.0, 0.0, 0.0),  # demo_patches carry no geotransform
    )

    classification = classify_task(request.query)
    try:
        plan = plan_from_query(request.query, scene)
        validate(plan)
        _, exec_trace = run_plan(plan, metadata, mask=mask, classes=classes)
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
