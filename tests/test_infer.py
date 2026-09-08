"""Tests for perception/infer.py.

Uses the real models/satquery_model.onnx + class_config.json with
synthetic (random) 16-channel input arrays for the integration-level
checks, and a deterministic fake ONNX session for precisely verifying the
tiling/feathered-blending math itself, independent of the real model's
black-box behavior.
"""

import numpy as np
import pytest

from perception import infer
from perception.infer import ClassConfig, InferenceError, InferenceResult

RNG = np.random.default_rng(0)


@pytest.fixture(scope="module")
def config() -> ClassConfig:
    return infer.load_class_config()


@pytest.fixture(scope="module")
def session(config):
    return infer.load_model()


def _random_image(channels=16, height=120, width=120) -> np.ndarray:
    return (RNG.random((channels, height, width)) * 1000).astype(np.float32)


# --- class_config.json loading -----------------------------------------------


def test_load_class_config_matches_the_model(config):
    assert config.nclasses == 19
    assert len(config.labels) == 19
    assert len(config.use_ch) == 16
    assert config.mean.shape == (16,)
    assert config.std.shape == (16,)


def test_load_model_has_the_expected_input_output_names(session):
    input_names = {i.name for i in session.get_inputs()}
    output_names = {o.name for o in session.get_outputs()}
    assert input_names == {"image", "sensor_id"}
    assert output_names == {"segmentation", "classification"}


# --- segment_image against the real model, synthetic input ------------------


def test_single_tile_exact_size(session, config):
    image = _random_image(height=120, width=120)
    result = infer.segment_image(image, "optical", session=session, config=config)

    assert isinstance(result, InferenceResult)
    assert result.mask.shape == (120, 120)
    assert result.probabilities.shape == (19, 120, 120)
    assert result.mask.min() >= 0 and result.mask.max() < 19

    sums = result.probabilities.sum(axis=0)
    assert np.allclose(sums, 1.0, atol=1e-4)  # a real softmax, not raw logits


@pytest.mark.parametrize("sensor", ["optical", "sar", "fused"])
def test_all_three_sensor_ids_run(session, config, sensor):
    image = _random_image(height=120, width=120)
    result = infer.segment_image(image, sensor, session=session, config=config)
    assert result.mask.shape == (120, 120)


def test_larger_image_is_tiled_and_stitched(session, config):
    image = _random_image(height=250, width=300)
    result = infer.segment_image(image, "optical", session=session, config=config)

    assert result.mask.shape == (250, 300)
    assert result.probabilities.shape == (19, 250, 300)
    sums = result.probabilities.sum(axis=0)
    assert np.allclose(sums, 1.0, atol=1e-4)  # holds everywhere, including overlap zones


def test_image_smaller_than_one_tile_is_cropped_back(session, config):
    image = _random_image(height=60, width=80)
    result = infer.segment_image(image, "fused", session=session, config=config)

    assert result.mask.shape == (60, 80)          # not (120, 120) -- padding was cropped away
    assert result.probabilities.shape == (19, 60, 80)


def test_transform_is_passed_through_unchanged(session, config):
    sentinel_transform = object()
    result = infer.segment_image(
        _random_image(), "optical", session=session, config=config, transform=sentinel_transform
    )
    assert result.transform is sentinel_transform


def test_unknown_sensor_raises(session, config):
    with pytest.raises(InferenceError, match="sensor"):
        infer.segment_image(_random_image(), "lidar", session=session, config=config)


def test_insufficient_channels_raises(session, config):
    with pytest.raises(InferenceError, match="channel"):
        infer.segment_image(_random_image(channels=4), "optical", session=session, config=config)


def test_run_inference_convenience_loads_defaults():
    result = infer.run_inference(_random_image(), "optical")
    assert result.mask.shape == (120, 120)
    assert result.labels[0] == "Agro-forestry areas"


# --- tiling/blending internals, tested precisely with a fake session -------


class _FakeSession:
    """Ignores its input entirely; returns a strong, one-hot-like
    prediction for `target_class`, in call order -- lets us know exactly
    which tile produced which logits, to check the blend math precisely."""

    def __init__(self, class_sequence):
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


def test_feathered_blending_is_a_smooth_crossfade_not_a_seam():
    # height=210 with tile_size=120, stride=90 -> exactly two tiles,
    # (0:120) and (90:210), overlapping by exactly 30 rows (90:120).
    config = _uniform_config()
    fake_session = _FakeSession(class_sequence=[5, 7])  # tile0 -> class 5, tile1 -> class 7
    image = np.zeros((16, 210, 120), dtype=np.float32)

    result = infer.segment_image(image, "optical", session=fake_session, config=config)

    assert result.mask.shape == (210, 120)
    assert fake_session.calls == 2  # confirms exactly 2 tiles were run, as designed

    # far from the seam: each tile's own prediction wins outright
    assert result.mask[0, 0] == 5
    assert result.mask[209, 0] == 7

    # inside the 30-row overlap band (global rows 90-119): a genuine
    # cross-fade, not an abrupt cutover
    p_top = result.probabilities[:, 90, 0]
    p_mid = result.probabilities[:, 104, 0]
    p_bottom = result.probabilities[:, 119, 0]

    assert p_top[5] > p_top[7]        # near the top of the overlap: still tile0-favored
    assert p_bottom[7] > p_bottom[5]  # near the bottom: tile1-favored
    assert abs(p_mid[5] - p_mid[7]) < abs(p_top[5] - p_top[7])  # more balanced mid-way through

    # probabilities are still a real distribution everywhere, including the blend
    assert np.allclose(result.probabilities.sum(axis=0), 1.0, atol=1e-4)


def test_tile_starts_covers_the_dimension_without_gaps():
    starts = infer._tile_starts(size=210, tile_size=120, stride=90)
    assert starts[0] == 0
    assert starts[-1] + 120 == 210
    for start in starts:
        assert 0 <= start <= 210 - 120


def test_tile_starts_single_tile_when_image_not_larger_than_tile():
    assert infer._tile_starts(size=80, tile_size=120, stride=90) == [0]
    assert infer._tile_starts(size=120, tile_size=120, stride=90) == [0]


def test_feather_weight_is_full_strength_in_the_core_and_tapers_at_the_edges():
    weight = infer._feather_weight(tile_size=120, overlap_px=30)
    assert weight.shape == (120, 120)
    assert weight[60, 60] == pytest.approx(1.0)   # center: full weight
    assert weight[0, 0] < weight[30, 30]           # corner ramps up towards the core
    assert weight[0, 0] == pytest.approx((1 / 30) ** 2)


def test_extract_tile_pads_and_reports_the_valid_region():
    image = np.arange(16 * 50 * 40, dtype=np.float32).reshape(16, 50, 40)
    tile, valid_h, valid_w = infer._extract_tile(image, row0=0, col0=0, tile_size=120)

    assert tile.shape == (16, 120, 120)
    assert valid_h == 50 and valid_w == 40
    assert np.array_equal(tile[:, :50, :40], image)
    assert np.all(tile[:, 50:, :] == 0)  # padded region is zero, not leaked garbage
    assert np.all(tile[:, :, 40:] == 0)
