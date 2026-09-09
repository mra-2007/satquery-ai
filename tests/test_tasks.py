"""Tests for agent/tasks.py: task classification via a mocked Gemini
client, the disk cache, and the deterministic keyword fallback."""

import json

import pytest

from agent import tasks
from agent.tasks import (
    FEW_SHOT_EXAMPLES,
    TASK_LABELS,
    ClassificationResult,
    classify_task,
)
from evidence.schema import TraceStep


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text


class _FakeModels:
    def __init__(self, responses, calls):
        self._responses = list(responses)
        self._calls = calls

    def generate_content(self, *, model, contents, config=None):
        self._calls.append(contents)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(item)


class _FakeClient:
    def __init__(self, responses):
        self.calls: list[str] = []
        self.models = _FakeModels(responses, self.calls)


def _gemini_json(label: str, confidence: float = 0.9) -> str:
    return json.dumps({"label": label, "confidence": confidence})


# --- labels and few-shot prompt ----------------------------------------------


def test_eight_task_labels():
    assert TASK_LABELS == (
        "vqa", "caption", "grounding", "change_vqa", "change_describe", "cross_modal",
        "metadata", "conversational",
    )


def test_few_shot_examples_cover_every_label():
    assert {label for _, label in FEW_SHOT_EXAMPLES} == set(TASK_LABELS)


def test_prompt_includes_query_and_all_labels():
    prompt = tasks._build_prompt("How many buildings are there?")
    assert "How many buildings are there?" in prompt
    for label in TASK_LABELS:
        assert label in prompt


# --- classify_task against a mocked Gemini client ----------------------------


def test_classify_task_uses_gemini_response(tmp_path):
    client = _FakeClient([_gemini_json("vqa", 0.92)])
    result = classify_task("How many water bodies are there?", client=client, cache_dir=tmp_path)

    assert isinstance(result, ClassificationResult)
    assert result.label == "vqa"
    assert result.confidence == pytest.approx(0.92)
    assert len(client.calls) == 1


def test_label_appears_in_the_trace(tmp_path):
    client = _FakeClient([_gemini_json("caption", 0.85)])
    result = classify_task("Describe this image.", client=client, cache_dir=tmp_path)

    assert len(result.trace) == 1
    entry = result.trace[0]
    assert isinstance(entry, TraceStep)
    assert entry.task == "classify_task"
    assert entry.tool == "agent.tasks.classify_task"
    assert entry.output["label"] == "caption"
    assert entry.confidence == pytest.approx(0.85)


def test_cache_hit_never_touches_the_api(tmp_path):
    poison_client = _FakeClient([AssertionError("must not be called")])
    working_client = _FakeClient([_gemini_json("grounding", 0.8)])

    first = classify_task("Where is the largest building?", client=working_client, cache_dir=tmp_path)
    assert len(working_client.calls) == 1

    second = classify_task("Where is the largest building?", client=poison_client, cache_dir=tmp_path)
    assert second.label == first.label
    assert second.confidence == first.confidence
    assert len(poison_client.calls) == 0


def test_api_exception_falls_back_to_keyword_parser(tmp_path):
    client = _FakeClient([ConnectionError("network down")])
    result = classify_task("How many water bodies are there?", client=client, cache_dir=tmp_path)
    assert result.label == "vqa"
    assert result.trace[0].parameters["source"] == "keyword_fallback"


def test_malformed_response_falls_back_to_keyword_parser(tmp_path):
    client = _FakeClient(["not valid json at all"])
    result = classify_task("Describe what changed between these two images.", client=client, cache_dir=tmp_path)
    assert result.label == "change_describe"
    assert result.trace[0].parameters["source"] == "keyword_fallback"


def test_unknown_label_in_response_falls_back(tmp_path):
    client = _FakeClient([_gemini_json("not_a_real_label", 0.9)])
    result = classify_task("How many buildings are there?", client=client, cache_dir=tmp_path)
    assert result.label == "vqa"
    assert result.trace[0].parameters["source"] == "keyword_fallback"


def test_out_of_range_confidence_falls_back(tmp_path):
    client = _FakeClient([_gemini_json("vqa", 1.5)])
    result = classify_task("How many buildings are there?", client=client, cache_dir=tmp_path)
    assert result.trace[0].parameters["source"] == "keyword_fallback"


def test_no_api_key_and_no_client_uses_fallback_directly(tmp_path, monkeypatch):
    monkeypatch.setattr(tasks, "GOOGLE_API_KEY", None)
    result = classify_task("How many water bodies are there?", cache_dir=tmp_path)
    assert result.label == "vqa"
    assert result.trace[0].parameters["source"] == "keyword_fallback"


# --- keyword fallback, one per label ------------------------------------------


@pytest.mark.parametrize("query,expected", [
    ("How many water bodies are there?", "vqa"),
    ("Describe this satellite image.", "caption"),
    ("Where is the largest building?", "grounding"),
    ("Did the urban area grow between 2019 and 2023?", "change_vqa"),
    ("Describe what changed between these two images.", "change_describe"),
    ("Does the SAR image confirm the flooding seen in the optical image?", "cross_modal"),
    ("Where is this?", "metadata"),
    ("When was this taken?", "metadata"),
    ("What sensor is this?", "metadata"),
    ("What resolution is this?", "metadata"),
    ("What's in this scene?", "metadata"),
    ("Hi there!", "conversational"),
    ("Hello, how are you?", "conversational"),
])
def test_keyword_fallback_shapes(query, expected):
    label, confidence = tasks._keyword_fallback(query)
    assert label == expected
    assert 0.0 <= confidence <= 1.0


def test_keyword_fallback_default_has_lower_confidence():
    # No distinctive signal at all -- still labeled "vqa" here (a coarse
    # trace label only), but agent/planner.py's own, more detailed keyword
    # fallback is what actually decides whether a real plan can be built
    # for a query like this, independent of this label -- see
    # tests/test_planner.py's test_fallback_unparseable_query_is_conversational.
    label, confidence = tasks._keyword_fallback("What is the meaning of life?")
    assert label == "vqa"
    assert confidence < 0.75


def test_where_is_this_is_metadata_not_grounding():
    # "where is this" (the scene's own location) must not fall through to
    # "grounding" ("where is the X" -- locate a region within the scene)
    # just because both start with "where is".
    label, _confidence = tasks._keyword_fallback("Where is this scene located?")
    assert label == "metadata"
