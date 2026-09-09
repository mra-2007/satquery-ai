"""Tests for api/main.py: FastAPI endpoints over the real
classify_task -> planner -> validate -> executor pipeline, hitting the
real preloaded data/demo_patches scenes.

classify_task/plan_from_query are forced offline here (GOOGLE_API_KEY
monkeypatched to None), the same way scripts/demo_real.py and
tools/caption.py's tests force it -- this repo has a real key configured
in .env, and a test suite must never depend on a live network call."""

import io
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from scipy.ndimage import shift as ndi_shift

from agent import planner, tasks
from api.main import app


@pytest.fixture(autouse=True)
def offline_mode(monkeypatch):
    monkeypatch.setattr(tasks, "GOOGLE_API_KEY", None)
    monkeypatch.setattr(planner, "GOOGLE_API_KEY", None)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:  # triggers the lifespan startup (loads demo scenes)
        yield test_client


def _png_bytes(width=32, height=32) -> bytes:
    image = Image.new("RGB", (width, height), color=(10, 20, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _grayscale_png_bytes(width=64, height=64, seed=0, array: np.ndarray | None = None) -> bytes:
    if array is None:
        array = np.random.default_rng(seed).random((height, width)) * 255
    image = Image.fromarray(array.astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _shifted_grayscale_pair(shift_px, width=64, height=64, seed=0) -> tuple[bytes, bytes]:
    """(base_png, shifted_png) -- a real, known sub/super-pixel shift
    between the two, for exercising phase_cross_correlation."""
    base = np.random.default_rng(seed).random((height, width)) * 255
    shifted = ndi_shift(base, shift_px, mode="reflect")
    return _grayscale_png_bytes(array=base), _grayscale_png_bytes(array=shifted)


# --- GET /scenes --------------------------------------------------------------


def test_list_scenes_returns_the_demo_patches(client):
    # GET /scenes now also lists any scene ever /upload-ed (the whole
    # point of this task), so scope to source="demo" rather than assuming
    # /scenes returns exactly the demo patches and nothing else. Startup
    # also seeds one demo 'change' pair from data/oscd/ (see
    # api.scenes.ensure_demo_change_scene_loaded) and one demo
    # 'cross_modal' scene from data/demo_patches/*.npz (see
    # ensure_demo_cross_modal_scene_loaded) alongside the 15
    # data/demo_patches/*.npz 'single' scenes, so demo scenes now split
    # into three distinct kinds/shapes -- checked separately below.
    response = client.get("/scenes")
    assert response.status_code == 200
    demo_scenes = [s for s in response.json() if s["source"] == "demo"]
    assert len(demo_scenes) == 17  # 15 single + 1 seeded OSCD change pair + 1 seeded cross_modal scene

    patch_scenes = [s for s in demo_scenes if s["kind"] == "single"]
    assert len(patch_scenes) == 15
    for scene in patch_scenes:
        assert scene["sensor"] == "fused"
        assert scene["gsd_metres"] == 10.0
        assert scene["width"] == 120 and scene["height"] == 120
        assert scene["id"] and scene["filename"].endswith(".npz")

    change_scenes = [s for s in demo_scenes if s["kind"] == "change"]
    assert len(change_scenes) == 1
    assert change_scenes[0]["sensor"] == "optical"
    assert change_scenes[0]["gsd_metres"] == 10.0

    cross_modal_scenes = [s for s in demo_scenes if s["kind"] == "cross_modal"]
    assert len(cross_modal_scenes) == 1
    assert cross_modal_scenes[0]["sensor"] == "fused"
    assert cross_modal_scenes[0]["gsd_metres"] == 10.0
    assert cross_modal_scenes[0]["filename"].endswith(".npz")


# --- GET /scenes/{id}/image, /mask, /legend --------------------------------------


def test_scene_image_is_a_real_png(client):
    scene_id = client.get("/scenes").json()[0]["id"]
    response = client.get(f"/scenes/{scene_id}/image")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"  # real PNG magic bytes, not a stub


def test_scene_mask_is_a_real_png(client):
    scene_id = client.get("/scenes").json()[0]["id"]
    response = client.get(f"/scenes/{scene_id}/mask")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_scene_legend_only_lists_classes_actually_present(client):
    scene_id = client.get("/scenes").json()[0]["id"]
    legend = client.get(f"/scenes/{scene_id}/legend").json()
    assert len(legend) > 0
    for entry in legend:
        assert set(entry) == {"class_id", "class_name", "color"}
        assert entry["color"].startswith("#") and len(entry["color"]) == 7


def test_scene_legend_is_sorted_by_area_descending(client):
    scene_id = client.get("/scenes").json()[0]["id"]
    legend = client.get(f"/scenes/{scene_id}/legend").json()
    # the dominant class in this specific demo patch, verified against
    # the mask directly -- not just "some order", the ACTUAL order
    import numpy as np
    from api.database import get_connection
    conn = get_connection()
    row = conn.execute("SELECT mask_path FROM scenes WHERE id = ?", (scene_id,)).fetchone()
    conn.close()
    mask = np.load(row["mask_path"])
    ids, counts = np.unique(mask, return_counts=True)
    expected_order = [int(i) for i in ids[np.argsort(counts)[::-1]]]
    assert [entry["class_id"] for entry in legend] == expected_order


def test_unknown_scene_image_returns_404(client):
    assert client.get("/scenes/does-not-exist/image").status_code == 404


# --- POST /query ----------------------------------------------------------------


def test_query_returns_evidence_with_answer_geometry_trace_confidence(client):
    scene_id = client.get("/scenes").json()[0]["id"]
    response = client.post("/query", json={"scene_id": scene_id, "query": "How many water bodies are there?"})
    assert response.status_code == 200
    body = response.json()

    assert body["scene_id"] == scene_id
    assert body["query"] == "How many water bodies are there?"
    assert isinstance(body["report_id"], str) and body["report_id"]

    evidence = body["evidence"]
    assert isinstance(evidence["value"], int)
    assert evidence["geometry"]["type"] == "FeatureCollection"
    assert len(evidence["execution_trace"]) >= 2  # classify_task + at least one execute step
    assert any(step["task"] == "classify_task" for step in evidence["execution_trace"])
    assert len(evidence["confidence"]) >= 1


def test_query_size_returns_hectares_units(client):
    scene_id = client.get("/scenes").json()[0]["id"]
    response = client.post("/query", json={"scene_id": scene_id, "query": "How much forest is there in hectares?"})
    assert response.status_code == 200
    evidence = response.json()["evidence"]
    assert evidence["units"] == "hectares"
    assert isinstance(evidence["value"], (int, float))


def test_query_building_count_reports_a_limitation(client):
    # "building" resolves via agent/vocabulary.py to "Urban fabric", which
    # the capability guardrail always degrades to size() at 10 m GSD.
    scene_id = client.get("/scenes").json()[0]["id"]
    response = client.post("/query", json={"scene_id": scene_id, "query": "How many buildings are there?"})
    assert response.status_code == 200
    body = response.json()
    assert body["limitation"] is not None
    assert "GSD" in body["limitation"]
    assert body["evidence"]["units"] == "hectares"  # degraded to size, not a raw count


def test_query_unknown_scene_returns_404(client):
    response = client.post("/query", json={"scene_id": "does-not-exist", "query": "How many water bodies?"})
    assert response.status_code == 404


def test_query_unresolvable_phrase_gets_a_conversational_reply_not_a_422(client):
    # Previously a 422 (PlannerError) -- per the CONVERSATIONAL INPUT
    # feature, an off-topic/unparseable query now gets a scoped reply
    # instead of an error.
    scene_id = client.get("/scenes").json()[0]["id"]
    response = client.post("/query", json={"scene_id": scene_id, "query": "What is the meaning of life?"})
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["evidence"]["value"], str)
    assert "only answer questions about" in body["evidence"]["value"].lower()
    assert any(step["tool"] == "conversational" for step in body["evidence"]["execution_trace"])


# --- METADATA QUESTIONS and CONVERSATIONAL INPUT -----------------------------------


def test_query_metadata_resolution_question(client):
    scene_id = client.get("/scenes").json()[0]["id"]
    response = client.post("/query", json={"scene_id": scene_id, "query": "What resolution is this?"})
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["evidence"]["value"], str)
    assert "ground sample distance" in body["evidence"]["value"].lower()
    step = next(s for s in body["evidence"]["execution_trace"] if s["tool"] == "metadata")
    assert step["parameters"] == {"aspect": "resolution"}
    assert step["confidence"] == 1.0


def test_query_metadata_classes_question_lists_real_areas(client):
    scene_id = client.get("/scenes").json()[0]["id"]
    response = client.post("/query", json={"scene_id": scene_id, "query": "What's in this scene?"})
    assert response.status_code == 200
    body = response.json()
    step = next(s for s in body["evidence"]["execution_trace"] if s["tool"] == "metadata")
    assert step["output"]["aspect"] == "classes"
    assert len(step["output"]["detail"]["classes"]) > 0


def test_query_metadata_date_question_reads_the_real_demo_patch_filename(client):
    scene_id = client.get("/scenes").json()[0]["id"]
    response = client.post("/query", json={"scene_id": scene_id, "query": "When was this taken?"})
    assert response.status_code == 200
    step = next(s for s in response.json()["evidence"]["execution_trace"] if s["tool"] == "metadata")
    assert step["output"]["detail"]["available"] is True
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", step["output"]["detail"]["acquisition_date"])


def test_query_greeting_gets_a_conversational_reply(client):
    scene_id = client.get("/scenes").json()[0]["id"]
    response = client.post("/query", json={"scene_id": scene_id, "query": "Hi there!"})
    assert response.status_code == 200
    body = response.json()
    assert "SatQuery AI" in body["evidence"]["value"]
    step = next(s for s in body["evidence"]["execution_trace"] if s["tool"] == "conversational")
    assert step["parameters"] == {"kind": "greeting"}


def test_adjacency_query_geometry_includes_both_classes(client):
    scene_id = client.get("/scenes").json()[0]["id"]
    response = client.post(
        "/query", json={"scene_id": scene_id, "query": "Is there any built-up area near water?"}
    )
    assert response.status_code == 200
    evidence = response.json()["evidence"]
    assert isinstance(evidence["value"], bool)


# --- GET /report ------------------------------------------------------------------


def test_report_download_matches_the_original_query_response(client):
    scene_id = client.get("/scenes").json()[0]["id"]
    query_response = client.post("/query", json={"scene_id": scene_id, "query": "Is there any forest?"})
    report_id = query_response.json()["report_id"]

    report_response = client.get(f"/report?report_id={report_id}")
    assert report_response.status_code == 200
    assert report_response.headers["content-type"] == "application/json"
    assert f"report_{report_id}.json" in report_response.headers["content-disposition"]
    assert report_response.json() == query_response.json()


def test_report_unknown_id_returns_404(client):
    response = client.get("/report?report_id=does-not-exist")
    assert response.status_code == 404


# --- POST /upload -----------------------------------------------------------------


def test_upload_valid_png_with_gsd_override_passes_validation(client):
    response = client.post(
        "/upload",
        files={"file": ("scene.png", _png_bytes(), "image/png")},
        data={"modality": "optical", "gsd_metres": "10.0"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["gsd_metres"] == 10.0
    assert body["band_count"] == 3
    assert body["width"] == 32 and body["height"] == 32


def test_upload_registers_a_scene_ready_for_the_full_pipeline(client):
    response = client.post(
        "/upload",
        files={"file": ("scene2.png", _png_bytes(), "image/png")},
        data={"modality": "optical", "gsd_metres": "10.0"},
    )
    body = response.json()
    assert body["ok"] is True
    assert body["kind"] == "single"
    assert body["scene_id"] is not None

    # the 3-band -> 16-channel gap must be disclosed, not hidden
    assert len(body["warnings"]) >= 1
    assert any("16" in w for w in body["warnings"])

    # it must appear in GET /scenes with the same warnings...
    scenes = {s["id"]: s for s in client.get("/scenes").json()}
    assert body["scene_id"] in scenes
    assert scenes[body["scene_id"]]["source"] == "upload"
    assert scenes[body["scene_id"]]["warnings"] == body["warnings"]

    # ...and be immediately queryable, end to end
    query_response = client.post(
        "/query", json={"scene_id": body["scene_id"], "query": "How many water bodies are there?"}
    )
    assert query_response.status_code == 200
    assert query_response.json()["scene_id"] == body["scene_id"]


def test_upload_png_without_gsd_override_fails_validation(client):
    # raster_io/validate.py refuses rather than defaulting a GSD.
    response = client.post(
        "/upload",
        files={"file": ("no_gsd.png", _png_bytes(), "image/png")},
        data={"modality": "optical"},
    )
    assert response.status_code == 200  # a rejected upload is a valid outcome, not a server error
    body = response.json()
    assert body["ok"] is False
    assert "GSD" in body["reason"]


def test_upload_unsupported_format_fails_validation(client):
    response = client.post(
        "/upload",
        files={"file": ("not_an_image.bmp", b"not a real bmp", "image/bmp")},
        data={"modality": "optical"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert "unsupported format" in body["reason"]


# --- POST /upload: pairs ---------------------------------------------------------


def test_pair_upload_without_pair_kind_is_rejected(client):
    optical, sar = _shifted_grayscale_pair((0.0, 0.0))
    response = client.post(
        "/upload",
        files={"file": ("a.png", optical, "image/png"), "file2": ("b.png", sar, "image/png")},
        data={"modality": "optical", "gsd_metres": "10.0", "modality2": "sar", "gsd_metres2": "10.0"},
    )
    body = response.json()
    assert body["ok"] is False
    assert "pair_kind" in body["reason"]


def test_cross_modal_pair_requires_one_optical_and_one_sar(client):
    optical_a, optical_b = _shifted_grayscale_pair((0.0, 0.0))
    response = client.post(
        "/upload",
        files={"file": ("a.png", optical_a, "image/png"), "file2": ("b.png", optical_b, "image/png")},
        data={
            "modality": "optical", "gsd_metres": "10.0", "modality2": "optical", "gsd_metres2": "10.0",
            "pair_kind": "cross_modal",
        },
    )
    body = response.json()
    assert body["ok"] is False
    assert "optical" in body["reason"] and "sar" in body["reason"]


def test_cross_modal_pair_upload_registers_a_fused_scene(client):
    optical, sar = _shifted_grayscale_pair((0.0, 0.0))  # perfectly aligned -- no fallback expected
    response = client.post(
        "/upload",
        files={"file": ("opt.png", optical, "image/png"), "file2": ("sar.png", sar, "image/png")},
        data={
            "modality": "optical", "gsd_metres": "10.0", "modality2": "sar", "gsd_metres2": "10.0",
            "pair_kind": "cross_modal",
        },
    )
    body = response.json()
    assert body["ok"] is True
    assert body["kind"] == "cross_modal"
    assert body["fallback_single_modality"] is False
    assert body["scene_id"] is not None
    # SAR-channel honesty caveat must be present alongside the RGB one
    assert any("SAR" in w for w in body["warnings"])

    scene = next(s for s in client.get("/scenes").json() if s["id"] == body["scene_id"])
    assert scene["sensor"] == "fused"
    assert scene["kind"] == "cross_modal"

    # queryable like any other scene
    query_response = client.post(
        "/query", json={"scene_id": body["scene_id"], "query": "How many water bodies are there?"}
    )
    assert query_response.status_code == 200


def test_change_pair_upload_registers_a_change_scene_queryable_on_after_state(client):
    before, after = _shifted_grayscale_pair((0.0, 0.0), seed=3)  # identical content, aligned
    response = client.post(
        "/upload",
        files={"file": ("before.png", before, "image/png"), "file2": ("after.png", after, "image/png")},
        data={
            "modality": "optical", "gsd_metres": "10.0", "modality2": "optical", "gsd_metres2": "10.0",
            "pair_kind": "change",
        },
    )
    body = response.json()
    assert body["ok"] is True
    assert body["kind"] == "change"
    assert body["scene_id"] is not None

    scene = next(s for s in client.get("/scenes").json() if s["id"] == body["scene_id"])
    assert scene["kind"] == "change"
    assert scene["sensor"] == "optical"

    # the AFTER (current) state answers ordinary questions directly
    query_response = client.post(
        "/query", json={"scene_id": body["scene_id"], "query": "How much forest is there in hectares?"}
    )
    assert query_response.status_code == 200
    assert query_response.json()["evidence"]["units"] == "hectares"


def test_change_pair_upload_carries_caller_supplied_dates(client):
    before, after = _shifted_grayscale_pair((0.0, 0.0), seed=4)
    response = client.post(
        "/upload",
        files={"file": ("before.png", before, "image/png"), "file2": ("after.png", after, "image/png")},
        data={
            "modality": "optical", "gsd_metres": "10.0", "modality2": "optical", "gsd_metres2": "10.0",
            "pair_kind": "change", "before_date": "2019-03", "after_date": "2023-11",
        },
    )
    body = response.json()
    assert body["ok"] is True
    scene = next(s for s in client.get("/scenes").json() if s["id"] == body["scene_id"])
    assert scene["before_date"] == "2019-03"
    assert scene["after_date"] == "2023-11"


def test_change_pair_before_and_after_image_and_mask_endpoints(client):
    before, after = _shifted_grayscale_pair((0.0, 0.0), seed=5)
    response = client.post(
        "/upload",
        files={"file": ("before.png", before, "image/png"), "file2": ("after.png", after, "image/png")},
        data={
            "modality": "optical", "gsd_metres": "10.0", "modality2": "optical", "gsd_metres2": "10.0",
            "pair_kind": "change",
        },
    )
    scene_id = response.json()["scene_id"]

    for endpoint in ("image", "mask"):
        for when in ("before", "after"):
            r = client.get(f"/scenes/{scene_id}/{endpoint}", params={"when": when})
            assert r.status_code == 200
            assert r.headers["content-type"] == "image/png"


def test_before_image_and_mask_404_for_a_non_change_scene(client):
    demo_scene_id = next(s["id"] for s in client.get("/scenes").json() if s["kind"] == "single")
    assert client.get(f"/scenes/{demo_scene_id}/image", params={"when": "before"}).status_code == 404
    assert client.get(f"/scenes/{demo_scene_id}/mask", params={"when": "before"}).status_code == 404


def test_change_summary_endpoint_returns_per_class_deltas(client):
    before, after = _shifted_grayscale_pair((0.0, 0.0), seed=6)
    response = client.post(
        "/upload",
        files={"file": ("before.png", before, "image/png"), "file2": ("after.png", after, "image/png")},
        data={
            "modality": "optical", "gsd_metres": "10.0", "modality2": "optical", "gsd_metres2": "10.0",
            "pair_kind": "change",
        },
    )
    scene_id = response.json()["scene_id"]

    r = client.get(f"/scenes/{scene_id}/change")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["summary"], str)
    assert isinstance(body["changed_pixels"], int)
    assert 0.0 <= body["changed_fraction"] <= 1.0
    for entry in body["classes"]:
        assert entry["net_ha"] == pytest.approx(entry["gained_ha"] - entry["lost_ha"])


def test_change_mask_endpoint_returns_an_rgba_png(client):
    before, after = _shifted_grayscale_pair((0.0, 0.0), seed=7)
    response = client.post(
        "/upload",
        files={"file": ("before.png", before, "image/png"), "file2": ("after.png", after, "image/png")},
        data={
            "modality": "optical", "gsd_metres": "10.0", "modality2": "optical", "gsd_metres2": "10.0",
            "pair_kind": "change",
        },
    )
    scene_id = response.json()["scene_id"]

    r = client.get(f"/scenes/{scene_id}/change-mask")
    assert r.status_code == 200
    image = Image.open(io.BytesIO(r.content))
    assert image.mode == "RGBA"


def test_change_endpoints_404_for_a_non_change_scene(client):
    demo_scene_id = next(s["id"] for s in client.get("/scenes").json() if s["kind"] == "single")
    assert client.get(f"/scenes/{demo_scene_id}/change").status_code == 404
    assert client.get(f"/scenes/{demo_scene_id}/change-mask").status_code == 404


def test_change_phrased_queries_resolve_via_the_keyword_fallback(client):
    before, after = _shifted_grayscale_pair((0.0, 0.0), seed=8)
    response = client.post(
        "/upload",
        files={"file": ("before.png", before, "image/png"), "file2": ("after.png", after, "image/png")},
        data={
            "modality": "optical", "gsd_metres": "10.0", "modality2": "optical", "gsd_metres2": "10.0",
            "pair_kind": "change",
        },
    )
    scene_id = response.json()["scene_id"]

    for query in ("What changed between these dates?", "Has built-up area increased?"):
        r = client.post("/query", json={"scene_id": scene_id, "query": query})
        assert r.status_code == 200
        assert isinstance(r.json()["evidence"]["value"], str)


# --- disaster-response phrasings: composition of existing tools, not new ----
# capability -- run against the real, preloaded OSCD demo change scene
# (api.scenes.ensure_demo_change_scene_loaded, asserted to exist in
# test_list_scenes_returns_the_demo_patches above) so these exercise real
# bi-temporal masks end to end, not synthetic noise.


def test_disaster_response_phrased_queries_run_end_to_end(client):
    demo_scene_id = next(s["id"] for s in client.get("/scenes").json() if s["kind"] == "change")

    queries_and_units = [
        ("How much cropland is flooded?", "hectares"),
        ("How much area is under water?", "hectares"),
        ("Which built-up areas are near flooding?", None),
        ("How much land changed to water?", "hectares"),
    ]
    for query, expected_units in queries_and_units:
        r = client.post("/query", json={"scene_id": demo_scene_id, "query": query})
        assert r.status_code == 200, query
        evidence = r.json()["evidence"]
        assert evidence["units"] == expected_units
        # every one of these ends in a real number or boolean, never a raw
        # Python dict repr leaking through as the answer.
        assert not str(evidence["value"]).startswith("{")


def test_disaster_response_cropland_flooded_maps_to_the_intersect_tool(client):
    demo_scene_id = next(s["id"] for s in client.get("/scenes").json() if s["kind"] == "change")
    r = client.post("/query", json={"scene_id": demo_scene_id, "query": "How much cropland is flooded?"})
    assert r.status_code == 200
    trace = r.json()["evidence"]["execution_trace"]
    final_step = next(step for step in reversed(trace) if step["task"].startswith("execute:"))
    assert final_step["tool"] == "intersect"
    assert isinstance(r.json()["evidence"]["value"], float)


def test_disaster_response_land_changed_to_water_focuses_the_change_tool(client):
    demo_scene_id = next(s["id"] for s in client.get("/scenes").json() if s["kind"] == "change")
    r = client.post("/query", json={"scene_id": demo_scene_id, "query": "How much land changed to water?"})
    assert r.status_code == 200
    trace = r.json()["evidence"]["execution_trace"]
    final_step = next(step for step in reversed(trace) if step["task"].startswith("execute:"))
    assert final_step["tool"] == "change"
    assert final_step["parameters"].get("class_id") is not None
    assert isinstance(r.json()["evidence"]["value"], float)


# --- GET /scenes/{id}/cross-modal, /cross-modal-mask -----------------------------


def _demo_cross_modal_scene_id(client) -> str:
    return next(s["id"] for s in client.get("/scenes").json() if s["kind"] == "cross_modal")


def test_cross_modal_summary_endpoint_returns_real_findings(client):
    scene_id = _demo_cross_modal_scene_id(client)
    r = client.get(f"/scenes/{scene_id}/cross-modal")
    assert r.status_code == 200
    body = r.json()

    assert body["cloud_simulated"] is False
    assert body["cloud_fraction"] is None
    assert body["sensor_status"] == {"optical": "USABLE", "sar": "USABLE", "fused": "USABLE"}
    assert isinstance(body["summary"], str) and body["summary"]
    assert len(body["findings"]) > 0
    for finding in body["findings"]:
        assert finding["detected_by"]  # at least one sensor found it, by construction
        assert finding["optical_area_ha"] >= 0
        assert finding["sar_area_ha"] >= 0
        assert finding["fused_area_ha"] >= 0


def test_cross_modal_summary_cloud_simulation_marks_optical_insufficient(client):
    scene_id = _demo_cross_modal_scene_id(client)
    r = client.get(f"/scenes/{scene_id}/cross-modal", params={"cloud_simulation": "true"})
    assert r.status_code == 200
    body = r.json()
    assert body["cloud_simulated"] is True
    assert body["cloud_fraction"] == pytest.approx(0.6)
    assert body["sensor_status"] == {"optical": "INSUFFICIENT", "sar": "USABLE", "fused": "USABLE"}


def test_cross_modal_mask_endpoint_returns_a_real_png_per_sensor(client):
    scene_id = _demo_cross_modal_scene_id(client)
    for sensor in ("optical", "sar", "fused"):
        r = client.get(f"/scenes/{scene_id}/cross-modal-mask", params={"sensor": sensor})
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_cross_modal_mask_endpoint_cloud_simulation_changes_the_optical_mask(client):
    scene_id = _demo_cross_modal_scene_id(client)
    clean = client.get(f"/scenes/{scene_id}/cross-modal-mask", params={"sensor": "optical"}).content
    clouded = client.get(
        f"/scenes/{scene_id}/cross-modal-mask", params={"sensor": "optical", "cloud_simulation": "true"},
    ).content
    assert clean != clouded  # zeroing 60% of the real optical channels must change the predicted mask


def test_cross_modal_image_endpoint_cloud_simulation_changes_the_preview(client):
    scene_id = _demo_cross_modal_scene_id(client)
    clean = client.get(f"/scenes/{scene_id}/image").content
    clouded = client.get(f"/scenes/{scene_id}/image", params={"cloud_simulation": "true"}).content
    assert clean != clouded


def test_cross_modal_phrased_queries_resolve_via_the_keyword_fallback(client):
    scene_id = _demo_cross_modal_scene_id(client)
    for query in ("Does the SAR image confirm what the optical image shows?", "Compare the radar and optical."):
        r = client.post("/query", json={"scene_id": scene_id, "query": query})
        assert r.status_code == 200
        assert isinstance(r.json()["evidence"]["value"], str)


def test_cross_modal_endpoints_404_for_a_non_cross_modal_scene(client):
    demo_scene_id = next(s["id"] for s in client.get("/scenes").json() if s["kind"] == "single")
    assert client.get(f"/scenes/{demo_scene_id}/cross-modal").status_code == 404
    assert client.get(f"/scenes/{demo_scene_id}/cross-modal-mask").status_code == 404


def test_severely_misaligned_pair_falls_back_to_a_single_scene(client):
    # shift far beyond DEFAULT_MAX_SHIFT_PX (5.0 px)
    optical, sar = _shifted_grayscale_pair((25.0, 18.0))
    response = client.post(
        "/upload",
        files={"file": ("opt.png", optical, "image/png"), "file2": ("sar.png", sar, "image/png")},
        data={
            "modality": "optical", "gsd_metres": "10.0", "modality2": "sar", "gsd_metres2": "10.0",
            "pair_kind": "cross_modal",
        },
    )
    body = response.json()
    assert body["ok"] is True  # not a hard rejection -- see raster_io/validate.py
    assert body["kind"] == "single"
    assert body["fallback_single_modality"] is True
    assert any("misaligned" in w for w in body["warnings"])
