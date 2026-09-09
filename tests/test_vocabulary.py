"""Tests for agent/vocabulary.py: SEGMENTATION_CLASSES's shape, and
resolve_noun()'s single-class and multi-class (list) resolution.

This file didn't exist before the "forest resolves to only one class"
bug -- see the many-to-many resolve_noun tests below, which are
regression tests for exactly that bug."""

from agent.vocabulary import CLASS_NAME_TO_ID, SEGMENTATION_CLASSES, resolve_noun


def test_segmentation_classes_has_19_unique_entries():
    assert len(SEGMENTATION_CLASSES) == 19
    assert len(set(SEGMENTATION_CLASSES)) == 19


def test_class_name_to_id_matches_segmentation_classes_order():
    for i, name in enumerate(SEGMENTATION_CLASSES):
        assert CLASS_NAME_TO_ID[name] == i


# --- single-class resolution (unchanged behaviour) --------------------------


def test_resolve_noun_exact_match():
    assert resolve_noun("pastures") == "Pastures"


def test_resolve_noun_singular_plural_tolerance():
    assert resolve_noun("building") == "Urban fabric"
    assert resolve_noun("buildings") == "Urban fabric"


def test_resolve_noun_substring_fallback():
    assert resolve_noun("a small residential building") == "Urban fabric"


def test_resolve_noun_specific_forest_type_stays_single():
    assert resolve_noun("coniferous forest") == "Coniferous forest"
    assert resolve_noun("broad-leaved forest") == "Broad-leaved forest"
    assert resolve_noun("mixed forest") == "Mixed forest"


def test_resolve_noun_specific_water_type_stays_single():
    assert resolve_noun("sea") == "Marine waters"
    assert resolve_noun("lake") == "Inland waters"


def test_resolve_noun_unresolvable_returns_none():
    assert resolve_noun("spaceship") is None


# --- many-to-many resolution: the bug fix -----------------------------------


def test_resolve_noun_forest_resolves_to_all_three_real_forest_classes():
    result = resolve_noun("forest")
    assert isinstance(result, list)
    assert set(result) == {"Broad-leaved forest", "Coniferous forest", "Mixed forest"}


def test_resolve_noun_forests_plural_also_resolves_to_the_list():
    assert set(resolve_noun("forests")) == {"Broad-leaved forest", "Coniferous forest", "Mixed forest"}


def test_resolve_noun_water_resolves_to_both_real_water_classes():
    result = resolve_noun("water")
    assert isinstance(result, list)
    assert set(result) == {"Inland waters", "Marine waters"}


def test_resolve_noun_water_body_also_resolves_to_the_list():
    assert set(resolve_noun("water body")) == {"Inland waters", "Marine waters"}


def test_resolve_noun_water_area_also_resolves_to_the_list():
    assert set(resolve_noun("water area")) == {"Inland waters", "Marine waters"}


# --- disaster-response additions: flooding and cropland ----------------------


def test_resolve_noun_flood_words_resolve_to_the_real_water_classes():
    for word in ("flood", "flooding", "flooded"):
        result = resolve_noun(word)
        assert isinstance(result, list)
        assert set(result) == {"Inland waters", "Marine waters"}


def test_resolve_noun_cropland_resolves_to_all_three_cultivation_classes():
    # Widened from a single "Arable land" mapping to the same
    # many-to-many pattern as "forest"/"water" -- a scene that's cropland
    # entirely of one subtype (e.g. all Permanent crops, no Arable land)
    # must not read as "no cropland".
    result = resolve_noun("cropland")
    assert isinstance(result, list)
    assert set(result) == {"Arable land", "Permanent crops", "Complex cultivation patterns"}


def test_resolve_noun_farmland_and_field_stay_single_arable_land():
    # Only "cropland" itself was widened -- its near-synonyms keep their
    # existing, narrower single-class mapping.
    assert resolve_noun("farmland") == "Arable land"
    assert resolve_noun("field") == "Arable land"
