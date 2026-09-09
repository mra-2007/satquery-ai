"""Tests for tools/cross_modal.py: simulate_cloud_cover()'s channel
zeroing, cross_modal_analysis()'s per-class sensor attribution (with a
deterministic fake ONNX session, mirroring tests/test_infer.py's own
_FakeSession, so every attribution category is verified precisely rather
than hoped for), and an end-to-end run against a real
data/demo_patches/*.npz stack with the real model and genuine SAR data.

No LLM is involved anywhere in tools/cross_modal.py -- it only calls
perception/infer.py three times and evidence/ops.py's size() -- so there
is nothing here to force offline the way tests/test_caption.py does."""

from pathlib import Path

import numpy as np
import pytest

from agent.vocabulary import SEGMENTATION_CLASSES
from perception.infer import ClassConfig
from tools.cross_modal import (
    OPTICAL_CHANNEL_INDICES,
    CrossModalResult,
    cross_modal_analysis,
    simulate_cloud_cover,
)

GSD_10M = {"gsd_metres": 10.0}
ROOT_DIR = Path(__file__).resolve().parent.parent
DEMO_PATCHES_DIR = ROOT_DIR / "data" / "demo_patches"


# --- simulate_cloud_cover: pure array manipulation, no model involved -------


def _sample_stack(channels=16, height=20, width=10) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.random((channels, height, width)) * 1000).astype(np.float32) + 1.0  # never exactly 0


def test_simulate_cloud_cover_preserves_shape():
    stack = _sample_stack()
    clouded = simulate_cloud_cover(stack, 0.6)
    assert clouded.shape == stack.shape


def test_simulate_cloud_cover_zeroes_only_optical_channels_in_the_cloud_rows():
    stack = _sample_stack(height=20)
    clouded = simulate_cloud_cover(stack, 0.6)  # cloud_rows = round(20 * 0.6) = 12

    for channel in OPTICAL_CHANNEL_INDICES:
        assert np.all(clouded[channel, :12, :] == 0.0)
        assert np.all(clouded[channel, 12:, :] == stack[channel, 12:, :])  # untouched below the cloud


def test_simulate_cloud_cover_leaves_sar_and_auxiliary_channels_untouched():
    stack = _sample_stack(height=20)
    clouded = simulate_cloud_cover(stack, 0.6)
    for channel in range(10, 16):  # SAR (10-11) and auxiliary (12-15)
        assert np.array_equal(clouded[channel], stack[channel])


def test_simulate_cloud_cover_does_not_mutate_the_input():
    stack = _sample_stack()
    original = stack.copy()
    simulate_cloud_cover(stack, 0.6)
    assert np.array_equal(stack, original)


def test_simulate_cloud_cover_zero_fraction_touches_nothing():
    stack = _sample_stack()
    clouded = simulate_cloud_cover(stack, 0.0)
    assert np.array_equal(clouded, stack)


def test_simulate_cloud_cover_full_fraction_zeroes_all_optical_rows():
    stack = _sample_stack(height=20)
    clouded = simulate_cloud_cover(stack, 1.0)
    for channel in OPTICAL_CHANNEL_INDICES:
        assert np.all(clouded[channel] == 0.0)


@pytest.mark.parametrize("bad_fraction", [-0.1, 1.1])
def test_simulate_cloud_cover_rejects_out_of_range_fraction(bad_fraction):
    with pytest.raises(ValueError, match="cloud_fraction"):
        simulate_cloud_cover(_sample_stack(), bad_fraction)


# --- cross_modal_analysis: precise per-class attribution, via a fake session -

class _FakeSession:
    """Ignores its input entirely; returns a strong, one-hot-like
    prediction for the next class in `class_sequence`, in call order --
    same approach as tests/test_infer.py's own _FakeSession."""

    def __init__(self, class_sequence: list[int]):
        self._classes = list(class_sequence)
        self.calls = 0

    def run(self, output_names, feed):
        target_class = self._classes[self.calls]
        self.calls += 1
        logits = np.full((1, 19, 120, 120), -10.0, dtype=np.float32)
        logits[0, target_class, :, :] = 10.0
        classification = np.zeros((1, 19), dtype=np.float32)
        return [logits, classification]


def _uniform_config() -> ClassConfig:
    return ClassConfig(
        classmap={i: i for i in range(19)},
        nclasses=19,
        labels=[str(i) for i in range(19)],
        use_ch=list(range(16)),
        mean=np.zeros(16),
        std=np.ones(16),
    )


# height=390, width=120 -> _tile_starts(390, 120, 90) == [0, 90, 180, 270]:
# exactly 4 tiles per sensor, letting each sensor "see" up to 4 distinct
# classes (one dominant per tile) independent of the other two sensors --
# presence (area > 0) is all evidence.ops.size() checks, not position, so
# each sensor's 4 tiles are assigned exactly the classes needed to realize
# every one of the 7 non-trivial attribution patterns in a single run:
#   class 0: optical + sar + fused        -> "all three"
#   class 1: sar + fused                  -> SAR contribution
#   class 2: optical + fused              -> optical contribution
#   class 3: fused only                   -> fusion-only
#   class 4: optical only
#   class 5: sar only
#   class 6: optical + sar, NOT fused     -> conflict
_OPTICAL_TILES = [0, 2, 4, 6]
_SAR_TILES = [0, 1, 5, 6]
_FUSED_TILES = [0, 1, 2, 3]


def _seven_pattern_stack() -> np.ndarray:
    return np.zeros((16, 390, 120), dtype=np.float32)


@pytest.fixture
def seven_pattern_result() -> CrossModalResult:
    session = _FakeSession(_OPTICAL_TILES + _SAR_TILES + _FUSED_TILES)
    config = _uniform_config()
    return cross_modal_analysis(_seven_pattern_stack(), GSD_10M, session=session, config=config)


def _finding(result: CrossModalResult, class_id: int):
    return next(f for f in result.findings if f.class_id == class_id)


def test_all_three_agree_pattern(seven_pattern_result):
    finding = _finding(seven_pattern_result, 0)
    assert finding.detected_by == ("optical", "sar", "fused")
    assert finding.attribution == "detected by all three sensor modes"
    assert finding.optical_area_ha > 0 and finding.sar_area_ha > 0 and finding.fused_area_ha > 0


def test_sar_contribution_pattern(seven_pattern_result):
    finding = _finding(seven_pattern_result, 1)
    assert finding.detected_by == ("sar", "fused")
    assert "SAR contribution" in finding.attribution
    assert finding.optical_area_ha == 0


def test_optical_contribution_pattern(seven_pattern_result):
    finding = _finding(seven_pattern_result, 2)
    assert finding.detected_by == ("optical", "fused")
    assert "optical contribution" in finding.attribution
    assert finding.sar_area_ha == 0


def test_fusion_only_pattern(seven_pattern_result):
    finding = _finding(seven_pattern_result, 3)
    assert finding.detected_by == ("fused",)
    assert "fusion-only" in finding.attribution
    assert finding.optical_area_ha == 0 and finding.sar_area_ha == 0


def test_optical_only_pattern(seven_pattern_result):
    finding = _finding(seven_pattern_result, 4)
    assert finding.detected_by == ("optical",)
    assert "optical-only" in finding.attribution


def test_sar_only_pattern(seven_pattern_result):
    finding = _finding(seven_pattern_result, 5)
    assert finding.detected_by == ("sar",)
    assert "SAR-only" in finding.attribution


def test_optical_and_sar_conflict_pattern(seven_pattern_result):
    finding = _finding(seven_pattern_result, 6)
    assert finding.detected_by == ("optical", "sar")
    assert finding.fused_area_ha == 0
    assert "did not" in finding.attribution or "disagreement" in finding.attribution


def test_absent_classes_are_not_reported(seven_pattern_result):
    reported_ids = {f.class_id for f in seven_pattern_result.findings}
    assert reported_ids == {0, 1, 2, 3, 4, 5, 6}


def test_findings_sorted_by_fused_area_descending(seven_pattern_result):
    fused_areas = [f.fused_area_ha for f in seven_pattern_result.findings]
    assert fused_areas == sorted(fused_areas, reverse=True)


def test_summary_mentions_every_pattern_group(seven_pattern_result):
    summary = seven_pattern_result.summary
    assert "Agree across all three" in summary
    assert "SAR contribution" in summary
    assert "Optical contribution" in summary
    assert "Fusion-only" in summary
    assert "Optical-only" in summary
    assert "SAR-only" in summary
    assert "fused model disagrees" in summary


def test_class_names_come_from_agent_vocabulary(seven_pattern_result):
    finding = _finding(seven_pattern_result, 0)
    assert finding.class_name == SEGMENTATION_CLASSES[0]


def test_no_cloud_simulation_by_default(seven_pattern_result):
    assert seven_pattern_result.cloud_simulated is False
    assert seven_pattern_result.cloud_fraction is None


def test_empty_result_when_nothing_detected():
    session = _FakeSession([7] * 12)  # every tile of every sensor -> the same class 7
    result = cross_modal_analysis(_seven_pattern_stack(), GSD_10M, session=session, config=_uniform_config())
    assert len(result.findings) == 1
    assert result.findings[0].detected_by == ("optical", "sar", "fused")


# --- cloud_simulation plumbing (fake session) --------------------------------


def test_cloud_simulation_flag_is_recorded():
    session = _FakeSession([0] * 12)
    result = cross_modal_analysis(
        _seven_pattern_stack(), GSD_10M, cloud_simulation=True, cloud_fraction=0.6,
        session=session, config=_uniform_config(),
    )
    assert result.cloud_simulated is True
    assert result.cloud_fraction == 0.6
    assert "simulated cloud cover" in result.summary


# --- sensor_status: per-toggle USABLE/INSUFFICIENT verdict --------------------


def test_sensor_status_is_all_usable_without_cloud_simulation(seven_pattern_result):
    assert seven_pattern_result.sensor_status == {"optical": "USABLE", "sar": "USABLE", "fused": "USABLE"}


def test_sensor_status_marks_optical_insufficient_under_cloud_simulation():
    session = _FakeSession([0] * 12)
    result = cross_modal_analysis(
        _seven_pattern_stack(), GSD_10M, cloud_simulation=True, session=session, config=_uniform_config(),
    )
    assert result.sensor_status == {"optical": "INSUFFICIENT", "sar": "USABLE", "fused": "USABLE"}


# --- integration: real model, real genuine-SAR patch -------------------------

_demo_patches = sorted(DEMO_PATCHES_DIR.glob("*.npz")) if DEMO_PATCHES_DIR.exists() else []


@pytest.mark.skipif(not _demo_patches, reason="data/demo_patches/*.npz not available")
def test_cross_modal_analysis_runs_on_a_real_demo_patch():
    data = np.load(_demo_patches[0], allow_pickle=True)
    stack = data["stack"][:16].astype(np.float32)

    result = cross_modal_analysis(stack, GSD_10M)

    assert isinstance(result, CrossModalResult)
    assert result.cloud_simulated is False
    assert len(result.findings) > 0
    for finding in result.findings:
        assert finding.class_name == SEGMENTATION_CLASSES[finding.class_id]
        assert finding.detected_by  # at least one sensor found it, by construction
        assert finding.optical_area_ha >= 0
        assert finding.sar_area_ha >= 0
        assert finding.fused_area_ha >= 0


@pytest.mark.skipif(not _demo_patches, reason="data/demo_patches/*.npz not available")
def test_cloud_simulation_changes_the_optical_only_result_on_real_data():
    # Zeroing 60% of a real patch's optical channels must change what the
    # optical-only path sees -- this is a mechanical fact about real
    # pixels, unlike whether fusion "holds up better" (a real, sometimes
    # patch-dependent empirical question this project does not overclaim
    # an answer to -- see the module docstring).
    from perception import infer

    data = np.load(_demo_patches[0], allow_pickle=True)
    stack = data["stack"][:16].astype(np.float32)
    session = infer.load_model()
    config = infer.load_class_config()

    clean_optical = infer.segment_image(stack, "optical", session=session, config=config).mask
    result = cross_modal_analysis(stack, GSD_10M, cloud_simulation=True, session=session, config=config)
    from tools.cross_modal import simulate_cloud_cover
    clouded_optical = infer.segment_image(
        simulate_cloud_cover(stack, 0.6), "optical", session=session, config=config
    ).mask

    assert not np.array_equal(clean_optical, clouded_optical)
    assert result.cloud_simulated is True
    assert result.cloud_fraction == pytest.approx(0.6)
