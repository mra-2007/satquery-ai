"""Tests for tools/verifier.py: claim extraction from generated text, and
re-testing each claim against a hand-built mask via evidence/ops.py."""

import numpy as np
import pytest

from tools.verifier import verify_answer

GSD_10M = {"gsd_metres": 10.0}

URBAN_FABRIC = 0  # SEGMENTATION_CLASSES[0]
ARABLE_LAND = 2  # SEGMENTATION_CLASSES[2]
BROAD_LEAVED_FOREST = 8  # SEGMENTATION_CLASSES[8]
INLAND_WATERS = 17  # SEGMENTATION_CLASSES[17]


def _mask() -> np.ndarray:
    """10x10: mostly arable land (87 px = 0.87 ha), a 3x3 urban block (9 px
    = 0.09 ha, touching arable land), and a 2x2 forest block (4 px = 0.04
    ha, NOT touching urban). No water anywhere."""
    mask = np.full((10, 10), ARABLE_LAND, dtype=int)
    mask[0:3, 0:3] = URBAN_FABRIC
    mask[7:9, 7:9] = BROAD_LEAVED_FOREST
    return mask


# --- size claims --------------------------------------------------------------


def test_correct_size_claim_passes_unchanged():
    answer = "Urban fabric covers 0.09 ha (1 distinct area(s))."
    result = verify_answer(_mask(), GSD_10M, answer)
    assert result.all_passed
    assert result.verified_answer == answer
    assert len(result.claims) == 2  # size + count, from the combined caption shape
    size_claim = next(c for c in result.claims if c.claim_type == "size")
    assert size_claim.tool == "evidence.ops.size"
    assert size_claim.class_name == "Urban fabric"
    assert size_claim.recomputed_value == pytest.approx(0.09)
    assert size_claim.passed


def test_hallucinated_size_is_flagged_not_silently_kept():
    answer = "Urban fabric covers 50.00 ha (1 distinct area(s))."
    result = verify_answer(_mask(), GSD_10M, answer)
    assert not result.all_passed
    size_claim = next(c for c in result.claims if c.claim_type == "size")
    assert not size_claim.passed
    assert size_claim.claimed_value == pytest.approx(50.0)
    assert size_claim.recomputed_value == pytest.approx(0.09)
    assert "50.00" in size_claim.reason and "0.09" in size_claim.reason
    assert "[UNVERIFIED" in result.verified_answer
    assert "50.00 ha" in result.verified_answer  # flagged, not deleted


def test_terse_ha_of_class_form_is_recognised():
    result = verify_answer(_mask(), GSD_10M, "0.09 ha of Urban fabric")
    assert len(result.claims) == 1
    assert result.claims[0].claim_type == "size"
    assert result.claims[0].passed


def test_size_claim_tolerates_small_rounding():
    # true area is 0.09 ha; a claimed 0.091 is within the 2% tolerance
    result = verify_answer(_mask(), GSD_10M, "Urban fabric covers 0.091 ha.")
    assert result.all_passed


# --- count claims --------------------------------------------------------------


def test_correct_count_claim_passes():
    result = verify_answer(_mask(), GSD_10M, "There are 1 Urban fabric.")
    assert result.all_passed
    assert result.claims[0].claim_type == "count"
    assert result.claims[0].recomputed_value == 1


def test_hallucinated_count_is_flagged():
    result = verify_answer(_mask(), GSD_10M, "There are 3 Urban fabric.")
    assert not result.all_passed
    claim = result.claims[0]
    assert claim.claimed_value == 3
    assert claim.recomputed_value == 1
    assert not claim.passed
    assert result.verified_answer == "There are 3 Urban fabric. [UNVERIFIED: claimed 3, recomputed 1]"


def test_bare_terse_count_form_is_recognised():
    # api/rendering.py's build_answer() own terse count format, e.g. "3 water bodies"
    result = verify_answer(_mask(), GSD_10M, "3 water bodies")
    assert len(result.claims) == 1
    claim = result.claims[0]
    assert claim.claim_type == "count"
    assert claim.claimed_value == 3
    assert claim.recomputed_value == 0  # no water anywhere in this mask
    assert not claim.passed


# --- presence claims ------------------------------------------------------------


def test_presence_claim_passes_when_true():
    result = verify_answer(_mask(), GSD_10M, "Urban fabric is present.")
    assert result.all_passed
    claim = result.claims[0]
    assert claim.claim_type == "presence"
    assert claim.recomputed_value is True


def test_presence_claim_fails_when_false():
    result = verify_answer(_mask(), GSD_10M, "There is water in this scene.")
    assert not result.all_passed
    claim = result.claims[0]
    assert claim.claim_type == "presence"
    assert claim.recomputed_value is False
    assert "not present" in claim.reason


# --- adjacency claims ------------------------------------------------------------


def test_adjacency_claim_passes_when_true():
    result = verify_answer(_mask(), GSD_10M, "Urban fabric borders arable land.")
    assert result.all_passed
    claim = result.claims[0]
    assert claim.claim_type == "adjacency"
    assert claim.recomputed_value is True


def test_adjacency_claim_fails_when_false():
    # urban and forest are both present but do not touch in this mask
    result = verify_answer(_mask(), GSD_10M, "Urban fabric borders forest.")
    assert not result.all_passed
    claim = result.claims[0]
    assert claim.recomputed_value is False
    assert "not adjacent" in claim.reason


def test_adjacency_extracts_every_semicolon_joined_pair_not_just_the_first():
    # tools/caption.py's own render_template() packs every adjacent pair
    # into one "; "-joined clause -- a real bug this test locks in the fix
    # for: only checking the first pair silently ignored the rest.
    answer = "Adjacent classes: Urban fabric borders arable land; Urban fabric borders forest."
    result = verify_answer(_mask(), GSD_10M, answer)
    adjacency_claims = [c for c in result.claims if c.claim_type == "adjacency"]
    assert len(adjacency_claims) == 2
    assert adjacency_claims[0].passed  # urban touches arable land
    assert not adjacency_claims[1].passed  # urban and forest are in opposite corners, not touching


def test_adjacency_near_phrasing_is_recognised():
    result = verify_answer(_mask(), GSD_10M, "Urban fabric is near arable land.")
    assert result.claims[0].claim_type == "adjacency"
    assert result.all_passed


# --- unresolvable classes -------------------------------------------------------


def test_unresolvable_class_is_flagged_with_a_clear_reason():
    result = verify_answer(_mask(), GSD_10M, "There are 3 volcanoes.")
    assert not result.all_passed
    claim = result.claims[0]
    assert claim.class_name is None
    assert claim.recomputed_value is None
    assert "volcanoes" in claim.reason
    assert "[UNVERIFIED" in result.verified_answer


# --- multi-class (generic noun) resolution --------------------------------------


def test_generic_noun_resolves_to_every_matching_real_class():
    # "forest" spans Broad-leaved/Coniferous/Mixed forest (agent/vocabulary.py) --
    # this mask only has Broad-leaved, so size() must still find it via the union.
    result = verify_answer(_mask(), GSD_10M, "Forest covers 0.04 ha.")
    assert result.all_passed
    assert result.claims[0].class_name == "Broad-leaved forest / Coniferous forest / Mixed forest"


# --- on_unverifiable="drop" ------------------------------------------------------


def test_drop_mode_removes_the_failing_sentence_entirely():
    answer = "Urban fabric covers 0.09 ha (1 distinct area(s)). Urban fabric covers 50.00 ha (1 distinct area(s))."
    result = verify_answer(_mask(), GSD_10M, answer, on_unverifiable="drop")
    assert result.verified_answer == "Urban fabric covers 0.09 ha (1 distinct area(s))."


def test_drop_mode_can_empty_the_answer_entirely():
    result = verify_answer(_mask(), GSD_10M, "There are 3 volcanoes.", on_unverifiable="drop")
    assert result.verified_answer == ""


# --- non-claim prose is left alone ------------------------------------------------


def test_sentence_with_no_recognised_claim_passes_through_unchanged():
    result = verify_answer(_mask(), GSD_10M, "This is a satellite image of a mixed landscape.")
    assert result.claims == []
    assert result.all_passed  # vacuously true: nothing to fail
    assert result.verified_answer == "This is a satellite image of a mixed landscape."


def test_multi_sentence_mixed_answer():
    answer = (
        "This is a mixed landscape. "
        "Urban fabric covers 0.09 ha (1 distinct area(s)). "
        "There are 3 volcanoes."
    )
    result = verify_answer(_mask(), GSD_10M, answer)
    # the "covers ... distinct area(s)" sentence yields both a size and a
    # count claim; the volcano sentence yields one more (unresolvable) --
    # the plain descriptive first sentence yields none.
    assert len(result.claims) == 3
    assert result.all_passed is False
    assert result.verified_answer.startswith("This is a mixed landscape. Urban fabric covers 0.09 ha")
    assert "[UNVERIFIED" in result.verified_answer


# --- original_answer is preserved verbatim ----------------------------------------


def test_original_answer_is_never_mutated():
    answer = "Urban fabric covers 50.00 ha (1 distinct area(s))."
    result = verify_answer(_mask(), GSD_10M, answer)
    assert result.original_answer == answer
    assert result.verified_answer != answer
