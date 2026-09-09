"""Unit tests for api/scenes.py's channel-mapping helpers:
build_model_input_from_bands (optical) and place_sar_bands (SAR) -- the
honest, disclosed handling of an upload that doesn't have the model's
real 16 Sentinel-1/2 channels."""

import numpy as np

from api.scenes import RGB_TO_DN_SCALE, build_model_input_from_bands, place_sar_bands


def _data(band_count, height=8, width=6, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((band_count, height, width)) * 255).astype(np.uint8)


# --- build_model_input_from_bands: RGB (3-band) ------------------------------


def test_rgb_maps_into_the_real_b04_b03_b02_positions_not_a_naive_copy():
    data = _data(3, height=4, width=4)
    channels, warnings = build_model_input_from_bands(data)

    assert channels.shape == (16, 4, 4)
    # band 0 = R -> channel 2 (B04); band 1 = G -> channel 1 (B03); band 2 = B -> channel 0 (B02)
    assert np.allclose(channels[2], data[0].astype(np.float32) * RGB_TO_DN_SCALE)
    assert np.allclose(channels[1], data[1].astype(np.float32) * RGB_TO_DN_SCALE)
    assert np.allclose(channels[0], data[2].astype(np.float32) * RGB_TO_DN_SCALE)
    # NOT a naive channels[:3] = data copy, which would leave Red in channel 0
    assert not np.allclose(channels[0], data[0].astype(np.float32) * RGB_TO_DN_SCALE)


def test_rgb_zero_fills_the_remaining_13_channels():
    channels, _warnings = build_model_input_from_bands(_data(3))
    for channel in range(3, 16):
        if channel in (0, 1, 2):
            continue
        assert np.all(channels[channel] == 0.0)


def test_rgb_produces_a_disclosed_warning():
    _channels, warnings = build_model_input_from_bands(_data(3))
    assert len(warnings) == 1
    assert "16" in warnings[0] and "zero-filled" in warnings[0]


# --- build_model_input_from_bands: other band counts -------------------------


def test_single_band_is_placed_at_channel_zero_and_warns():
    data = _data(1)
    channels, warnings = build_model_input_from_bands(data)
    assert np.array_equal(channels[0], data[0].astype(np.float32))
    assert np.all(channels[1:] == 0.0)
    assert len(warnings) == 1
    assert "1" in warnings[0] and "16" in warnings[0]


def test_exactly_16_bands_pass_through_with_no_warning():
    data = _data(16)
    channels, warnings = build_model_input_from_bands(data)
    assert np.array_equal(channels, data.astype(np.float32))
    assert warnings == []


def test_more_than_16_bands_are_truncated_with_a_warning():
    data = _data(20)
    channels, warnings = build_model_input_from_bands(data)
    assert np.array_equal(channels, data[:16].astype(np.float32))
    assert len(warnings) == 1
    assert "20" in warnings[0]


# --- place_sar_bands -----------------------------------------------------------


def test_two_band_sar_fills_vv_and_vh():
    channels = np.zeros((16, 4, 4), dtype=np.float32)
    sar = _data(2, height=4, width=4)
    warnings = place_sar_bands(channels, sar)

    assert np.array_equal(channels[10], sar[0].astype(np.float32))
    assert np.array_equal(channels[11], sar[1].astype(np.float32))
    assert any("raw uploaded pixel intensities" in w for w in warnings)


def test_single_band_sar_fills_only_vv_and_warns():
    channels = np.zeros((16, 4, 4), dtype=np.float32)
    sar = _data(1, height=4, width=4)
    warnings = place_sar_bands(channels, sar)

    assert np.array_equal(channels[10], sar[0].astype(np.float32))
    assert np.all(channels[11] == 0.0)
    assert any("only 1 channel" in w for w in warnings)


def test_sar_values_are_not_rescaled_unlike_optical_rgb():
    # No principled dB scale exists for arbitrary uploaded pixels -- SAR
    # values must be used as-is, not multiplied by RGB_TO_DN_SCALE.
    channels = np.zeros((16, 4, 4), dtype=np.float32)
    sar = _data(2, height=4, width=4)
    place_sar_bands(channels, sar)
    assert np.array_equal(channels[10], sar[0].astype(np.float32))
    assert not np.allclose(channels[10], sar[0].astype(np.float32) * RGB_TO_DN_SCALE)
