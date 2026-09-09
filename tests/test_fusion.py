"""Tests for evidence/fusion.py: the spectral-index formulas in isolation,
has_real_sar()'s zero-channel detection, each of the four evidence
sources individually (via a deterministic fake ONNX session, mirroring
tests/test_infer.py's and tests/test_cross_modal.py's own fake-session
pattern), and -- the central point of this module -- that the weighting
scheme actually changes the fused verdict versus a naive unweighted vote.

No LLM is involved anywhere in evidence/fusion.py, so there is nothing here
to force offline the way tests/test_caption.py does."""

from pathlib import Path

import numpy as np
import pytest

from agent.vocabulary import SEGMENTATION_CLASSES
from evidence import fusion
from evidence.fusion import FusionResult
from perception.infer import ClassConfig

GSD_10M = {"gsd_metres": 10.0}
ROOT_DIR = Path(__file__).resolve().parent.parent
DEMO_PATCHES_DIR = ROOT_DIR / "data" / "demo_patches"

INLAND_WATERS = SEGMENTATION_CLASSES.index("Inland waters")
URBAN_FABRIC = SEGMENTATION_CLASSES.index("Urban fabric")
MOORS = SEGMENTATION_CLASSES.index("Moors, heathland and sclerophyllous vegetation")


def _blank_stack(height=120, width=120) -> np.ndarray:
    return np.zeros((16, height, width), dtype=np.float32)


# --- has_real_sar: zero-channel detection -------------------------------------


def test_has_real_sar_false_for_zero_filled_channels():
    assert fusion.has_real_sar(_blank_stack()) is False


def test_has_real_sar_true_when_both_vv_and_vh_are_nonzero():
    stack = _blank_stack()
    stack[fusion.SAR_VV] = -12.0
    stack[fusion.SAR_VH] = -18.0
    assert fusion.has_real_sar(stack) is True


def test_has_real_sar_false_when_only_one_channel_is_nonzero():
    # place_sar_bands (api/scenes.py) can leave VH zero-filled for a
    # single-band SAR upload -- that's still not "real" dual-pol SAR.
    stack = _blank_stack()
    stack[fusion.SAR_VV] = -12.0
    assert fusion.has_real_sar(stack) is False


# --- spectral index formulas, hand-computed ------------------------------------


def test_ndvi_positive_for_healthy_vegetation_spectrum():
    stack = _blank_stack(height=1, width=1)
    stack[fusion.RED, 0, 0] = 500.0
    stack[fusion.NIR, 0, 0] = 2500.0
    assert fusion.ndvi(stack)[0, 0] == pytest.approx((2500 - 500) / (2500 + 500))


def test_ndwi_positive_for_open_water_spectrum():
    stack = _blank_stack(height=1, width=1)
    stack[fusion.GREEN, 0, 0] = 800.0
    stack[fusion.NIR, 0, 0] = 200.0
    assert fusion.ndwi(stack)[0, 0] == pytest.approx((800 - 200) / (800 + 200))


def test_ndbi_positive_for_built_up_spectrum():
    stack = _blank_stack(height=1, width=1)
    stack[fusion.SWIR1, 0, 0] = 1800.0
    stack[fusion.NIR, 0, 0] = 1200.0
    assert fusion.ndbi(stack)[0, 0] == pytest.approx((1800 - 1200) / (1800 + 1200))


def test_normalized_difference_handles_zero_denominator_without_nan():
    stack = _blank_stack(height=1, width=1)  # every band 0 -> denom 0
    assert fusion.ndvi(stack)[0, 0] == 0.0
    assert fusion.ndwi(stack)[0, 0] == 0.0
    assert fusion.ndbi(stack)[0, 0] == 0.0


# --- fake ONNX session: sensor-id-aware, so fused vs sar can disagree ---------


class _SensorAwareFakeSession:
    """Returns a strong, one-hot-like segmentation for whichever class
    `class_by_sensor_id` maps the call's sensor_id to (0=optical, 1=sar,
    2=fused), and a caller-controlled classification vector -- lets a test
    set the segmentation head's fused-sensor answer independently of its
    sar-sensor answer, which a plain class-sequence fake session (as used
    elsewhere) can't do since fusion calls segment_image twice with
    different sensor ids, not tile-by-tile in a fixed order."""

    def __init__(self, class_by_sensor_id: dict[int, int], cls_logits: np.ndarray | None = None):
        self._class_by_sensor_id = class_by_sensor_id
        self._cls_logits = cls_logits if cls_logits is not None else np.zeros(19, dtype=np.float32)

    def run(self, output_names, feed):
        sensor_id = int(feed["sensor_id"][0])
        target_class = self._class_by_sensor_id[sensor_id]
        logits = np.full((1, 19, 120, 120), -10.0, dtype=np.float32)
        logits[0, target_class, :, :] = 10.0
        classification = self._cls_logits.reshape(1, 19).astype(np.float32)
        return [logits, classification]


def _config_with_real_labels() -> ClassConfig:
    """Mirrors the REAL models/class_config.json: an alphabetically-sorted
    label order that is deliberately NOT SEGMENTATION_CLASSES's own order
    -- see agent/vocabulary.py's module docstring and this project's
    segmentation-vs-classification-label-order lesson. Using a genuinely
    different order here (not an identity/uniform stand-in) is the point:
    a test built on top of this config would fail if evidence/fusion.py
    ever looked up a classification probability by raw class_id instead of
    by class NAME."""
    return ClassConfig(
        classmap={i: i for i in range(19)},
        nclasses=19,
        labels=sorted(SEGMENTATION_CLASSES),
        use_ch=list(range(16)),
        mean=np.zeros(16),
        std=np.ones(16),
    )


def _one_hot_cls_logits(class_name: str, config: ClassConfig, high=8.0, low=-8.0) -> np.ndarray:
    logits = np.full(19, low, dtype=np.float32)
    logits[config.labels.index(class_name)] = high
    return logits


# --- individual sources, via the fake session ----------------------------------


def test_segmentation_evidence_reflects_the_fused_sensor_call():
    config = _config_with_real_labels()
    session = _SensorAwareFakeSession({2: INLAND_WATERS})
    stack = _blank_stack()

    result = fusion.fuse_evidence(stack, GSD_10M, INLAND_WATERS, session=session, config=config)
    seg = next(s for s in result.sources if s.source == "segmentation")

    assert seg.available is True
    assert seg.present is True
    assert seg.confidence > 0.9  # near-one-hot logits -> softmax close to 1.0


def test_classification_evidence_uses_the_configs_own_label_order():
    # The regression test for the label-order bug: cls_logits is one-hot
    # at "Inland waters"'s position in the CLASSIFICATION head's own
    # (alphabetical) order, which is NOT index INLAND_WATERS in
    # SEGMENTATION_CLASSES's order. If _classification_evidence ever
    # looked up by raw class_id instead of by name, this would read the
    # wrong (near-zero) logit and fail.
    config = _config_with_real_labels()
    cls_logits = _one_hot_cls_logits("Inland waters", config)
    session = _SensorAwareFakeSession({2: URBAN_FABRIC}, cls_logits=cls_logits)  # segmentation disagrees on purpose
    stack = _blank_stack()

    result = fusion.fuse_evidence(stack, GSD_10M, INLAND_WATERS, session=session, config=config)
    cls = next(s for s in result.sources if s.source == "classification")

    assert cls.available is True
    assert cls.present is True
    assert cls.confidence > 0.99  # sigmoid(8) ~= 0.9997


def test_classification_evidence_raises_loudly_for_an_unknown_class_name():
    config = ClassConfig(
        classmap={i: i for i in range(19)}, nclasses=19,
        labels=[f"unrelated-{i}" for i in range(19)],  # shares no names with SEGMENTATION_CLASSES at all
        use_ch=list(range(16)), mean=np.zeros(16), std=np.ones(16),
    )
    session = _SensorAwareFakeSession({2: INLAND_WATERS})
    with pytest.raises(ValueError, match="not found"):
        fusion.fuse_evidence(_blank_stack(), GSD_10M, INLAND_WATERS, session=session, config=config)


def test_spectral_evidence_unavailable_for_a_class_with_no_standard_index():
    config = _config_with_real_labels()
    session = _SensorAwareFakeSession({2: MOORS})
    result = fusion.fuse_evidence(_blank_stack(), GSD_10M, MOORS, session=session, config=config)
    spectral = next(s for s in result.sources if s.source == "spectral_index")
    assert spectral.available is False
    assert spectral.present is None and spectral.confidence is None


def test_spectral_evidence_present_when_ndwi_supports_water():
    config = _config_with_real_labels()
    session = _SensorAwareFakeSession({2: INLAND_WATERS})
    stack = _blank_stack()
    stack[fusion.GREEN, :, :] = 800.0
    stack[fusion.NIR, :, :] = 200.0  # NDWI > 0 everywhere -> supports water

    result = fusion.fuse_evidence(stack, GSD_10M, INLAND_WATERS, session=session, config=config)
    spectral = next(s for s in result.sources if s.source == "spectral_index")

    assert spectral.available is True
    assert spectral.present is True
    # NDWI = (800-200)/(800+200) = 0.6 everywhere -> peak 0.6, rescaled to (0.6+1)/2
    assert spectral.confidence == pytest.approx(0.8)


def test_spectral_evidence_absent_when_ndwi_contradicts_water():
    config = _config_with_real_labels()
    session = _SensorAwareFakeSession({2: INLAND_WATERS})
    stack = _blank_stack()
    stack[fusion.GREEN, :, :] = 200.0
    stack[fusion.NIR, :, :] = 800.0  # NDWI < 0 everywhere -> contradicts water

    result = fusion.fuse_evidence(stack, GSD_10M, INLAND_WATERS, session=session, config=config)
    spectral = next(s for s in result.sources if s.source == "spectral_index")

    assert spectral.available is True
    assert spectral.present is False
    # NDWI = (200-800)/(200+800) = -0.6 everywhere -> peak -0.6, rescaled to (-0.6+1)/2
    assert spectral.confidence == pytest.approx(0.2)


def test_sar_evidence_unavailable_without_real_sar():
    config = _config_with_real_labels()
    session = _SensorAwareFakeSession({2: INLAND_WATERS, 1: INLAND_WATERS})
    result = fusion.fuse_evidence(_blank_stack(), GSD_10M, INLAND_WATERS, session=session, config=config)
    sar = next(s for s in result.sources if s.source == "sar")
    assert sar.available is False
    assert sar.present is None and sar.confidence is None


def test_sar_evidence_available_and_independent_of_the_fused_call():
    config = _config_with_real_labels()
    # fused (sensor_id=2) says Urban fabric; sar (sensor_id=1) says Inland
    # waters -- the two calls disagree on purpose, to prove sar reads its
    # OWN segment_image call, not the fused one's mask.
    session = _SensorAwareFakeSession({2: URBAN_FABRIC, 1: INLAND_WATERS})
    stack = _blank_stack()
    stack[fusion.SAR_VV], stack[fusion.SAR_VH] = -12.0, -18.0

    result = fusion.fuse_evidence(stack, GSD_10M, INLAND_WATERS, session=session, config=config)
    sar = next(s for s in result.sources if s.source == "sar")
    seg = next(s for s in result.sources if s.source == "segmentation")

    assert sar.available is True
    assert sar.present is True
    assert seg.present is False


# --- the central point: weighting changes the outcome -------------------------


def test_spectral_and_sar_are_weighted_higher_than_the_correlated_heads():
    assert fusion.SOURCE_WEIGHTS["spectral_index"] > fusion.SOURCE_WEIGHTS["segmentation"]
    assert fusion.SOURCE_WEIGHTS["spectral_index"] > fusion.SOURCE_WEIGHTS["classification"]
    assert fusion.SOURCE_WEIGHTS["sar"] > fusion.SOURCE_WEIGHTS["segmentation"]
    assert fusion.SOURCE_WEIGHTS["sar"] > fusion.SOURCE_WEIGHTS["classification"]


def test_correlated_sources_are_named_segmentation_and_classification():
    assert set(fusion.CORRELATED_SOURCES) == {"segmentation", "classification"}


def test_independent_sources_outvote_the_two_correlated_heads():
    # Segmentation AND classification (both correlated, weight 1 each)
    # agree the class IS present; spectral index AND SAR (both
    # independent, weight 2 each) both contradict them. A naive unweighted
    # vote ties 2-2; the weighted fusion must side with the independent
    # sources (weight 4 vs 2), landing on ABSENT. This is the whole point
    # of the weighting scheme this module exists to apply.
    config = _config_with_real_labels()
    cls_logits = _one_hot_cls_logits("Inland waters", config)
    session = _SensorAwareFakeSession({2: INLAND_WATERS, 1: URBAN_FABRIC}, cls_logits=cls_logits)
    stack = _blank_stack()
    stack[fusion.GREEN, :, :] = 200.0
    stack[fusion.NIR, :, :] = 800.0  # NDWI contradicts water
    stack[fusion.SAR_VV], stack[fusion.SAR_VH] = -12.0, -18.0  # real SAR, but segments Urban fabric, not water

    result = fusion.fuse_evidence(stack, GSD_10M, INLAND_WATERS, session=session, config=config)

    seg = next(s for s in result.sources if s.source == "segmentation")
    cls = next(s for s in result.sources if s.source == "classification")
    spectral = next(s for s in result.sources if s.source == "spectral_index")
    sar = next(s for s in result.sources if s.source == "sar")
    assert seg.present is True and cls.present is True
    assert spectral.present is False and sar.present is False

    assert result.fused_present is False  # independent sources win despite being outnumbered 2-to-2
    assert result.agreement_fraction == pytest.approx(0.5)  # exactly 2 of 4 sources agree with the fused verdict


def test_all_four_sources_agreeing_gives_full_agreement_fraction():
    config = _config_with_real_labels()
    cls_logits = _one_hot_cls_logits("Inland waters", config)
    session = _SensorAwareFakeSession({2: INLAND_WATERS, 1: INLAND_WATERS}, cls_logits=cls_logits)
    stack = _blank_stack()
    stack[fusion.GREEN, :, :] = 800.0
    stack[fusion.NIR, :, :] = 200.0
    stack[fusion.SAR_VV], stack[fusion.SAR_VH] = -12.0, -18.0

    result = fusion.fuse_evidence(stack, GSD_10M, INLAND_WATERS, session=session, config=config)

    assert result.fused_present is True
    assert result.agreement_fraction == pytest.approx(1.0)
    assert all(s.present is True for s in result.sources)


def test_weighting_note_discloses_the_correlation_and_is_carried_on_every_result():
    config = _config_with_real_labels()
    session = _SensorAwareFakeSession({2: INLAND_WATERS})
    result = fusion.fuse_evidence(_blank_stack(), GSD_10M, INLAND_WATERS, session=session, config=config)

    assert "encoder" in result.weighting_note
    assert "correlated" in result.weighting_note
    assert "spectral" in result.weighting_note.lower()
    assert "sar" in result.weighting_note.lower()


def test_agreement_dict_records_every_sources_own_verdict():
    config = _config_with_real_labels()
    session = _SensorAwareFakeSession({2: INLAND_WATERS})
    result = fusion.fuse_evidence(_blank_stack(), GSD_10M, INLAND_WATERS, session=session, config=config)

    assert set(result.agreement) == {"segmentation", "classification", "spectral_index", "sar"}
    assert result.agreement["segmentation"] is True
    assert result.agreement["sar"] is None  # blank stack has no real SAR -- unavailable, not "absent"


def test_class_name_and_summary_are_populated():
    config = _config_with_real_labels()
    session = _SensorAwareFakeSession({2: INLAND_WATERS})
    result = fusion.fuse_evidence(_blank_stack(), GSD_10M, INLAND_WATERS, session=session, config=config)

    assert result.class_name == "Inland waters"
    assert isinstance(result, FusionResult)
    assert "Inland waters" in result.summary


def test_class_id_list_unions_segmentation_and_sar_sources():
    # "water" resolves to [Inland waters, Marine waters] -- see
    # agent/vocabulary.py. A scene predicted as Marine waters must still
    # count as "present" for this list, the same union semantics
    # evidence/ops.py's own tools use.
    marine_waters = SEGMENTATION_CLASSES.index("Marine waters")
    config = _config_with_real_labels()
    session = _SensorAwareFakeSession({2: marine_waters})
    result = fusion.fuse_evidence(
        _blank_stack(), GSD_10M, [INLAND_WATERS, marine_waters], session=session, config=config,
    )
    seg = next(s for s in result.sources if s.source == "segmentation")
    assert seg.present is True
    assert result.class_name == "Inland waters / Marine waters"


# --- integration: real model, real genuine-SAR patch --------------------------

_demo_patches = sorted(DEMO_PATCHES_DIR.glob("*.npz")) if DEMO_PATCHES_DIR.exists() else []


@pytest.mark.skipif(not _demo_patches, reason="data/demo_patches/*.npz not available")
def test_fuse_evidence_runs_on_a_real_demo_patch():
    data = np.load(_demo_patches[0], allow_pickle=True)
    stack = data["stack"][:16].astype(np.float32)

    result = fusion.fuse_evidence(stack, GSD_10M, INLAND_WATERS)

    assert isinstance(result, FusionResult)
    # this project's demo patches carry genuine Sentinel-1 SAR (see
    # tools/cross_modal.py's own docstring) -- sar must be available
    sar = next(s for s in result.sources if s.source == "sar")
    assert sar.available is True
    for source in result.sources:
        if source.available:
            assert 0.0 <= source.confidence <= 1.0
    assert 0.0 <= result.fused_confidence <= 1.0
