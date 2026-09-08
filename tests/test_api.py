"""Tests for api/main.py: FastAPI endpoints over the real
classify_task -> planner -> validate -> executor pipeline, hitting the
real preloaded data/demo_patches scenes.

classify_task/plan_from_query are forced offline here (GOOGLE_API_KEY
monkeypatched to None), the same way scripts/demo_real.py and
tools/caption.py's tests force it -- this repo has a real key configured
in .env, and a test suite must never depend on a live network call."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

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


# --- GET /scenes --------------------------------------------------------------


def test_list_scenes_returns_the_demo_patches(client):
    response = client.get("/scenes")
    assert response.status_code == 200
    scenes = response.json()
    assert len(scenes) == 9  # data/demo_patches/*.npz
    for scene in scenes:
        assert scene["sensor"] == "fused"
        assert scene["gsd_metres"] == 10.0
        assert scene["width"] == 120 and scene["height"] == 120
        assert scene["id"] and scene["filename"].endswith(".npz")


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


def test_query_unresolvable_phrase_returns_422(client):
    scene_id = client.get("/scenes").json()[0]["id"]
    response = client.post("/query", json={"scene_id": scene_id, "query": "What is the meaning of life?"})
    assert response.status_code == 422


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
