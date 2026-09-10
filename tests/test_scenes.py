"""Unit tests for api/scenes.py's channel-mapping helpers:
build_model_input_from_bands (optical) and place_sar_bands (SAR) -- the
honest, disclosed handling of an upload that doesn't have the model's
real 16 Sentinel-1/2 channels. Also covers ensure_demo_change_scene_loaded's
own defensive guard for a deployment image that doesn't ship data/oscd/."""

import numpy as np

from api import scenes
from api.database import get_connection
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


# --- ensure_demo_change_scene_loaded: missing data/oscd/ must not crash ----
# startup -- a lean deployment image (see the Dockerfile) only ships
# data/demo_patches/, deliberately not the much larger data/oscd/.
#
# tests/conftest.py's session-scoped DB is shared across the whole test
# run, and other modules (test_api.py's `client` fixture, elsewhere in the
# same session) legitimately seed a real 'demo'+'change' scene against the
# REAL data/oscd/ before this file's own tests ever run -- so these tests
# must not assume the table starts empty, only that it ends up in the same
# state it started in (this function skipped, added nothing new).


def _demo_change_scene_count() -> int:
    conn = get_connection()
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM scenes WHERE source = 'demo' AND kind = 'change'"
        ).fetchone()[0]
    finally:
        conn.close()


def test_ensure_demo_change_scene_loaded_skips_gracefully_without_oscd(monkeypatch, tmp_path):
    monkeypatch.setattr(scenes, "OSCD_DIR", tmp_path / "oscd_missing")  # no such directory at all
    before = _demo_change_scene_count()

    scenes.ensure_demo_change_scene_loaded()  # must not raise

    assert _demo_change_scene_count() == before  # nothing new was seeded -- correctly skipped


def test_ensure_demo_change_scene_loaded_skips_when_oscd_dir_exists_but_parquet_is_missing(monkeypatch, tmp_path):
    # The directory itself existing isn't enough -- the specific parquet
    # split file is what's actually read.
    empty_oscd_dir = tmp_path / "oscd_empty"
    empty_oscd_dir.mkdir()
    monkeypatch.setattr(scenes, "OSCD_DIR", empty_oscd_dir)

    scenes.ensure_demo_change_scene_loaded()  # must not raise
