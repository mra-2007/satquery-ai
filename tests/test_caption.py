"""Tests for tools/caption.py: deterministic fact extraction from a
class-ID mask (evidence/ops.py, over agent/vocabulary.py's 19-class
vocabulary), template rendering, and Gemini smoothing behind a mocked
client -- no real API calls."""

import numpy as np
import pytest

from tools import caption
from tools.caption import ClassFact

GSD_10M = {"gsd_metres": 10.0}

URBAN_FABRIC = 0  # SEGMENTATION_CLASSES[0]
ARABLE_LAND = 2  # SEGMENTATION_CLASSES[2]
CONIFEROUS_FOREST = 9  # SEGMENTATION_CLASSES[9]


def _mask() -> np.ndarray:
    """10x10, mostly arable land, a 3x3 urban block (touches arable land)
    in one corner and a 2x2 conifer block (also touches arable land, but
    is far from the urban block) in the opposite corner."""
    mask = np.full((10, 10), ARABLE_LAND, dtype=int)
    mask[0:3, 0:3] = URBAN_FABRIC  # 9 px = 0.09 ha
    mask[7:9, 7:9] = CONIFEROUS_FOREST  # 4 px = 0.04 ha
    return mask


class _FakeResponse:
    def __init__(self, text):
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


# --- extract_facts: the only function that reads the mask's areas ----------


def test_extract_facts_only_includes_present_classes_sorted_by_area():
    facts = caption.extract_facts(_mask(), GSD_10M)
    assert [f.class_id for f in facts] == [ARABLE_LAND, URBAN_FABRIC, CONIFEROUS_FOREST]


def test_extract_facts_areas_are_computed_by_evidence_ops():
    facts = caption.extract_facts(_mask(), GSD_10M)
    by_id = {f.class_id: f for f in facts}
    assert by_id[ARABLE_LAND].area_ha == pytest.approx(0.87)  # (100-9-4) px * 100 m2 / 10_000
    assert by_id[URBAN_FABRIC].area_ha == pytest.approx(0.09)
    assert by_id[CONIFEROUS_FOREST].area_ha == pytest.approx(0.04)


def test_extract_facts_fragment_counts():
    facts = caption.extract_facts(_mask(), GSD_10M)
    by_id = {f.class_id: f for f in facts}
    assert by_id[URBAN_FABRIC].fragment_count == 1
    assert by_id[CONIFEROUS_FOREST].fragment_count == 1


def test_extract_facts_single_class_mask_returns_one_fact():
    mask = np.full((5, 5), ARABLE_LAND, dtype=int)
    facts = caption.extract_facts(mask, GSD_10M)
    assert len(facts) == 1
    assert facts[0].class_id == ARABLE_LAND
    assert facts[0].area_ha == pytest.approx(0.25)  # 25 px * 100 m2 / 10_000


# --- extract_adjacent_pairs: same tool as every other adjacency question ---


def test_extract_adjacent_pairs_finds_touching_classes():
    facts = caption.extract_facts(_mask(), GSD_10M)
    pairs = {frozenset(p) for p in caption.extract_adjacent_pairs(_mask(), GSD_10M, facts)}
    assert frozenset({"Urban fabric", "Arable land"}) in pairs
    assert frozenset({"Coniferous forest", "Arable land"}) in pairs


def test_extract_adjacent_pairs_excludes_classes_that_are_far_apart():
    facts = caption.extract_facts(_mask(), GSD_10M)
    pairs = {frozenset(p) for p in caption.extract_adjacent_pairs(_mask(), GSD_10M, facts)}
    assert frozenset({"Urban fabric", "Coniferous forest"}) not in pairs


# --- render_template: pure formatting, no computation -----------------------


def test_render_template_no_classes():
    assert caption.render_template([], []) == "No land-cover classes were detected in this image."


def test_render_template_includes_dominant_class_area_and_fragment_count():
    fact = ClassFact(class_id=2, class_name="Arable land", area_ha=1.23, fragment_count=2)
    text = caption.render_template([fact], [])
    assert "Arable land" in text
    assert "1.23 ha" in text
    assert "2 distinct area(s)" in text


def test_render_template_mentions_every_present_class():
    facts = [
        ClassFact(2, "Arable land", 0.87, 1),
        ClassFact(0, "Urban fabric", 0.09, 1),
    ]
    text = caption.render_template(facts, [])
    assert "Arable land" in text
    assert "Urban fabric" in text


def test_render_template_includes_adjacency():
    facts = [ClassFact(0, "Urban fabric", 0.09, 1), ClassFact(2, "Arable land", 0.87, 1)]
    text = caption.render_template(facts, [("Urban fabric", "Arable land")])
    assert "Urban fabric borders Arable land" in text


def test_render_template_omits_adjacency_sentence_when_no_pairs_touch():
    facts = [ClassFact(2, "Arable land", 1.0, 1)]
    text = caption.render_template(facts, [])
    assert "borders" not in text


# --- caption(): end-to-end, facts always real, Gemini only rewords ---------


def test_caption_falls_back_to_template_with_no_client_and_no_api_key(monkeypatch):
    monkeypatch.setattr(caption, "GOOGLE_API_KEY", None)
    result = caption.caption(_mask(), GSD_10M)
    assert result.smoothed is False
    assert result.caption == result.template_sentence


def test_caption_uses_gemini_to_smooth_wording():
    reworded = "This scene is mostly arable land, with a small urban patch and an isolated conifer stand."
    client = _FakeClient([reworded])
    result = caption.caption(_mask(), GSD_10M, client=client)

    assert result.smoothed is True
    assert result.caption == reworded
    assert "Arable land" in result.template_sentence  # the fact-only sentence is still exposed
    assert len(client.calls) == 1


def test_caption_falls_back_when_gemini_raises():
    client = _FakeClient([ConnectionError("network down")])
    result = caption.caption(_mask(), GSD_10M, client=client)
    assert result.smoothed is False
    assert result.caption == result.template_sentence


def test_caption_falls_back_when_gemini_returns_empty_text():
    client = _FakeClient([""])
    result = caption.caption(_mask(), GSD_10M, client=client)
    assert result.smoothed is False
    assert result.caption == result.template_sentence


def test_caption_max_length_truncates_the_final_caption(monkeypatch):
    monkeypatch.setattr(caption, "GOOGLE_API_KEY", None)  # no client either -- must not hit the real API
    result = caption.caption(_mask(), GSD_10M, max_length=10)
    assert len(result.caption) == 10
    assert result.caption == result.template_sentence[:10]


def test_caption_without_max_length_is_not_truncated(monkeypatch):
    monkeypatch.setattr(caption, "GOOGLE_API_KEY", None)
    result = caption.caption(_mask(), GSD_10M)
    assert result.caption == result.template_sentence


def test_caption_facts_and_pairs_match_the_standalone_extraction_functions(monkeypatch):
    monkeypatch.setattr(caption, "GOOGLE_API_KEY", None)
    mask = _mask()
    expected_facts = caption.extract_facts(mask, GSD_10M)
    expected_pairs = caption.extract_adjacent_pairs(mask, GSD_10M, expected_facts)

    result = caption.caption(mask, GSD_10M)

    assert [f.class_id for f in result.facts] == [f.class_id for f in expected_facts]
    assert result.adjacent_pairs == expected_pairs
