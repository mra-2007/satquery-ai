"""Tests for agent/planner.py: prompt/few-shot sanity, the disk cache, the
deterministic keyword fallback, and plan_from_query's retry/fallback logic
against a mocked Gemini client (no real API calls)."""

import json

import pytest

from agent import planner
from agent.dsl import validate
from agent.planner import (
    FEW_SHOT_EXAMPLES,
    PlannerError,
    SceneDescriptor,
    plan_from_query,
)

TEST_SCENE = SceneDescriptor(
    layers=["sentinel2_10m_rgb", "class_raster"],
    classes={0: "land", 1: "water", 2: "forest", 3: "urban", 4: "agricultural"},
    sensor="sentinel-2",
    gsd_metres=10.0,
    bbox=(77.5, 12.9, 77.7, 13.1),
)


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


VALID_PLAN_JSON = json.dumps(
    [{"id": "s1", "tool": "count", "parameters": {"class_id": 1, "min_area_m2": 500}}]
)


# --- few-shot examples --------------------------------------------------------


def test_fifteen_few_shot_examples():
    assert len(FEW_SHOT_EXAMPLES) == 15


def test_few_shot_examples_cover_all_four_tools():
    tools_used = {step["tool"] for ex in FEW_SHOT_EXAMPLES for step in ex["plan"]}
    assert tools_used == {"presence", "count", "size", "adjacency"}


def test_every_few_shot_plan_is_itself_valid():
    from agent.dsl import Plan

    for example in FEW_SHOT_EXAMPLES:
        plan = Plan.model_validate(example["plan"])
        validate(plan)  # no raise


def test_prompt_includes_query_tools_and_error_feedback():
    prompt = planner._build_prompt(
        "How many water bodies?", TEST_SCENE,
        prior_plan_text="[bad]", prior_error="boom",
    )
    assert "How many water bodies?" in prompt
    assert "count" in prompt and "adjacency" in prompt
    assert "[bad]" in prompt
    assert "boom" in prompt


# --- keyword fallback ---------------------------------------------------------


def test_fallback_how_many():
    plan = planner._keyword_fallback("How many water bodies are there?", TEST_SCENE)
    assert plan[0].tool == "count"
    assert plan[0].parameters["class_id"] == 1


def test_fallback_how_much_area():
    plan = planner._keyword_fallback("How much area of forest?", TEST_SCENE)
    assert plan[0].tool == "size"
    assert plan[0].parameters["class_id"] == 2


def test_fallback_is_x_near_y():
    plan = planner._keyword_fallback("Is water near urban?", TEST_SCENE)
    assert plan[0].tool == "adjacency"
    assert plan[0].parameters == {"class_a": 1, "class_b": 3, "distance_m": 0.0}


def test_fallback_is_x_within_n_metres_of_y():
    plan = planner._keyword_fallback("Is water within 50 metres of urban?", TEST_SCENE)
    assert plan[0].tool == "adjacency"
    assert plan[0].parameters == {"class_a": 1, "class_b": 3, "distance_m": 50.0}


def test_fallback_presence():
    plan = planner._keyword_fallback("Is forest present?", TEST_SCENE)
    assert plan[0].tool == "presence"
    assert plan[0].parameters["class_id"] == 2


def test_fallback_how_much_x_is_there_without_the_word_area():
    plan = planner._keyword_fallback("How much forest is there in hectares?", TEST_SCENE)
    assert plan[0].tool == "size"
    assert plan[0].parameters["class_id"] == 2


def test_fallback_change_query_maps_to_the_parameterless_change_tool():
    plan = planner._keyword_fallback("What changed between these dates?", TEST_SCENE)
    assert plan[0].tool == "change"
    assert plan[0].parameters == {}


def test_fallback_change_query_matches_increase_phrasing_too():
    # No class-name parsing needed at all -- the change tool takes no
    # parameters, so "built-up area" never needs to resolve to a class_id.
    plan = planner._keyword_fallback("Has built-up area increased?", TEST_SCENE)
    assert plan[0].tool == "change"
    assert plan[0].parameters == {}


def test_fallback_cross_modal_query_maps_to_the_parameterless_cross_modal_tool():
    # No class-name parsing needed at all -- the cross_modal tool takes no
    # required parameters, so "SAR"/"radar" need only be recognized, not
    # resolved to a class_id.
    plan = planner._keyword_fallback("Does the SAR image confirm what the optical image shows?", TEST_SCENE)
    assert plan[0].tool == "cross_modal"
    assert plan[0].parameters == {}


def test_fallback_cross_modal_query_matches_radar_phrasing_too():
    plan = planner._keyword_fallback("Compare the radar and optical imagery for this area.", TEST_SCENE)
    assert plan[0].tool == "cross_modal"
    assert plan[0].parameters == {}


def test_fallback_describe_query_maps_to_the_parameterless_caption_tool():
    plan = planner._keyword_fallback("Describe this scene.", TEST_SCENE)
    assert plan[0].tool == "caption"
    assert plan[0].parameters == {}


def test_fallback_summarize_phrasing_also_maps_to_caption():
    plan = planner._keyword_fallback("Summarize what this image shows.", TEST_SCENE)
    assert plan[0].tool == "caption"


def test_resolve_class_matches_a_paraphrase_via_shared_prefix():
    scene = TEST_SCENE.model_copy(update={"classes": {**TEST_SCENE.classes, 3: "building"}})
    assert planner._resolve_class("there any built-up area", scene) == 3


def test_fallback_unparseable_query_is_a_conversational_off_topic_reply():
    # Previously raised PlannerError (surfaced as a 422 in api/main.py) --
    # per this feature's own requirement, an off-topic/unparseable query
    # now gets a scoped conversational reply instead of an error.
    plan = planner._keyword_fallback("What is the meaning of life?", TEST_SCENE)
    assert plan[0].tool == "conversational"
    assert plan[0].parameters == {"kind": "off_topic"}


def test_fallback_unknown_class_raises():
    # Unlike a fully unparseable query, this DID match a recognized
    # pattern (count) -- only the specific noun failed to resolve to a
    # known class -- so it still raises, not a conversational reply.
    with pytest.raises(PlannerError):
        planner._keyword_fallback("How many volcanoes are there?", TEST_SCENE)


# --- greetings and metadata questions ------------------------------------------


def test_fallback_greeting_maps_to_the_parameterless_conversational_tool():
    plan = planner._keyword_fallback("Hi there!", TEST_SCENE)
    assert plan[0].tool == "conversational"
    assert plan[0].parameters == {"kind": "greeting"}


@pytest.mark.parametrize("greeting", ["hi", "hello", "hey", "good morning", "howdy there"])
def test_fallback_recognizes_common_greetings(greeting):
    plan = planner._keyword_fallback(greeting, TEST_SCENE)
    assert plan[0].tool == "conversational"
    assert plan[0].parameters == {"kind": "greeting"}


@pytest.mark.parametrize("query,expected_aspect", [
    ("Where is this?", "location"),
    ("Where is this scene located?", "location"),
    ("When was this taken?", "date"),
    ("What is the acquisition date?", "date"),
    ("What sensor is this?", "sensor"),
    ("What platform captured this?", "sensor"),
    ("What resolution is this?", "resolution"),
    ("What is the ground sample distance?", "resolution"),
    ("What's in this scene?", "classes"),
    ("What is in this scene?", "classes"),
])
def test_fallback_metadata_questions_map_to_the_right_aspect(query, expected_aspect):
    plan = planner._keyword_fallback(query, TEST_SCENE)
    assert plan[0].tool == "metadata"
    assert plan[0].parameters == {"aspect": expected_aspect}


def test_fallback_where_is_this_does_not_fall_through_to_grounding():
    plan = planner._keyword_fallback("Where is this?", TEST_SCENE)
    assert plan[0].tool == "metadata"


def test_fallback_where_is_the_x_does_not_match_the_metadata_location_pattern():
    # "the water body" is not "this" -- must not be swallowed by the
    # metadata location pattern above it. (Offline grounding itself isn't
    # keyword-fallback-parseable at all -- a pre-existing gap, unrelated to
    # this feature -- so this still ends up a conversational off-topic
    # reply rather than a "ground" plan; the point of this test is only
    # that it ISN'T misrouted to "metadata".)
    plan = planner._keyword_fallback("Where is the water body?", TEST_SCENE)
    assert plan[0].tool != "metadata"


# --- plan_from_query: cache, retries, fallback --------------------------------


def test_cache_hit_never_touches_the_api(tmp_path):
    poison_client = _FakeClient([AssertionError("must not be called")])
    working_client = _FakeClient([VALID_PLAN_JSON])

    first = plan_from_query(
        "How many water bodies?", TEST_SCENE, client=working_client, cache_dir=tmp_path
    )
    assert len(working_client.calls) == 1

    second = plan_from_query(
        "How many water bodies?", TEST_SCENE, client=poison_client, cache_dir=tmp_path
    )
    assert second == first
    assert len(poison_client.calls) == 0  # cache hit -- API never touched


def test_first_try_success(tmp_path):
    client = _FakeClient([VALID_PLAN_JSON])
    plan = plan_from_query("Any water here?", TEST_SCENE, client=client, cache_dir=tmp_path)
    assert plan[0].tool == "count"
    assert len(client.calls) == 1


def test_retries_on_invalid_response_then_succeeds(tmp_path):
    bad = json.dumps([{"id": "s1", "tool": "levitate", "parameters": {}}])
    client = _FakeClient([bad, VALID_PLAN_JSON])

    plan = plan_from_query("Any water here?", TEST_SCENE, client=client, cache_dir=tmp_path)

    assert plan[0].tool == "count"
    assert len(client.calls) == 2
    # the second prompt must include the first attempt's error, fed back
    assert "levitate" in client.calls[1] or "unknown tool" in client.calls[1]


def test_exhausts_retries_and_raises(tmp_path):
    bad = json.dumps([{"id": "s1", "tool": "levitate", "parameters": {}}])
    client = _FakeClient([bad, bad])

    with pytest.raises(PlannerError):
        plan_from_query("Any water here?", TEST_SCENE, client=client, max_retries=2,
                         cache_dir=tmp_path)


def test_api_unavailable_falls_back_to_keyword_parser(tmp_path):
    client = _FakeClient([ConnectionError("network down")])

    plan = plan_from_query(
        "How many water bodies?", TEST_SCENE, client=client, cache_dir=tmp_path
    )

    assert plan[0].tool == "count"
    assert plan[0].parameters["class_id"] == 1
    # the fallback result was cached too
    assert plan_from_query(
        "How many water bodies?", TEST_SCENE,
        client=_FakeClient([AssertionError("must not be called")]), cache_dir=tmp_path,
    ) == plan


def test_no_api_key_and_no_client_uses_fallback_directly(tmp_path, monkeypatch):
    monkeypatch.setattr(planner, "GOOGLE_API_KEY", None)
    plan = plan_from_query("How many water bodies?", TEST_SCENE, cache_dir=tmp_path)
    assert plan[0].tool == "count"
    assert plan[0].parameters["class_id"] == 1


def test_cache_key_differs_by_scene(tmp_path):
    other_scene = TEST_SCENE.model_copy(update={"gsd_metres": 30.0})
    client_a = _FakeClient([VALID_PLAN_JSON])
    client_b = _FakeClient([VALID_PLAN_JSON])

    plan_from_query("Any water here?", TEST_SCENE, client=client_a, cache_dir=tmp_path)
    plan_from_query("Any water here?", other_scene, client=client_b, cache_dir=tmp_path)

    # different scenes -> different cache entries -> both clients were called
    assert len(client_a.calls) == 1
    assert len(client_b.calls) == 1
