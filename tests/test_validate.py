"""Tests for raster_io/validate.py: the pre-flight checks (count, format,
modality, GSD, band count, CRS) and pixel co-registration -- via
skimage.registration.phase_cross_correlation for a same-sensor pair, or
via matching dimensions/CRS alone for a declared cross-sensor
(`cross_sensor=True`) pair, where phase correlation isn't meaningful."""

from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import shift as ndi_shift

import raster_io.validate as validate_module
from raster_io.readers import RasterImage
from raster_io.validate import ImageInput, validate_images


def _image(
    path="a.tif", *, bands=1, size=64, gsd_metres=10.0, crs="EPSG:32643", seed=0, data=None,
) -> RasterImage:
    if data is None:
        data = np.random.default_rng(seed).random((bands, size, size)) * 255
        data = data.astype("uint8")
    return RasterImage(
        data=data, transform=None, crs=crs, gsd_metres=gsd_metres,
        band_count=data.shape[0], width=data.shape[2], height=data.shape[1], path=Path(path),
    )


# --- per-image checks ----------------------------------------------------------


def test_single_valid_image_passes():
    result = validate_images([ImageInput(_image(), "optical")])
    assert result.ok is True
    assert result.reason is None


def test_wrong_image_count_rejected():
    result = validate_images([ImageInput(_image(), "optical")], expected_count=2)
    assert result.ok is False
    assert "expected 2" in result.reason


def test_zero_images_rejected():
    result = validate_images([])
    assert result.ok is False
    assert "no images" in result.reason


def test_more_than_two_images_rejected():
    inputs = [ImageInput(_image(f"{i}.tif"), "optical") for i in range(3)]
    result = validate_images(inputs)
    assert result.ok is False


def test_unsupported_format_rejected():
    result = validate_images([ImageInput(_image("weird.bmp"), "optical")])
    assert result.ok is False
    assert "unsupported format" in result.reason


def test_wrong_modality_rejected():
    result = validate_images([ImageInput(_image(), "sar")], expected_modality="optical")
    assert result.ok is False
    assert "modality" in result.reason


def test_none_gsd_is_refused_not_defaulted():
    result = validate_images([ImageInput(_image(gsd_metres=None), "optical")])
    assert result.ok is False
    assert "no resolvable GSD" in result.reason


def test_gsd_outside_expected_range_rejected():
    result = validate_images(
        [ImageInput(_image(gsd_metres=100.0), "optical")], expected_gsd_range=(1.0, 60.0)
    )
    assert result.ok is False
    assert "GSD" in result.reason


def test_insufficient_band_count_rejected():
    result = validate_images([ImageInput(_image(bands=1), "optical")], min_band_count=3)
    assert result.ok is False
    assert "band(s)" in result.reason


# --- CRS check for a pair -------------------------------------------------------


def test_mismatched_crs_rejected():
    a = _image("a.tif", crs="EPSG:32643", size=32)
    b = _image("b.tif", crs="EPSG:32644", size=32)
    result = validate_images([ImageInput(a, "optical"), ImageInput(b, "optical")])
    assert result.ok is False
    assert "different CRS" in result.reason


def test_require_matching_crs_false_skips_the_crs_check_entirely():
    # both missing CRS (typical for two plain PNG/JPEG uploads) -- would
    # normally be a hard rejection; opting out relies on phase correlation
    # itself as the alignment check instead.
    a = _image("a.tif", crs=None, size=32)
    b = _image("b.tif", crs=None, size=32, data=a.data.copy())
    result = validate_images(
        [ImageInput(a, "optical"), ImageInput(b, "sar")], require_matching_crs=False,
    )
    assert result.ok is True
    assert result.fallback_single_modality is False


def test_require_matching_crs_false_still_runs_coregistration():
    a = _image("a.tif", crs=None, size=64)
    from scipy.ndimage import shift as _shift
    shifted = _shift(a.data[0].astype(float), (2.0, -1.0), mode="reflect")
    b = _image("b.tif", crs=None, size=64, data=shifted[None, :, :].astype("uint8"))

    result = validate_images(
        [ImageInput(a, "optical"), ImageInput(b, "sar")], require_matching_crs=False,
    )
    assert result.ok is True
    assert result.auto_shifted is True
    assert result.shift_px is not None


def test_missing_crs_on_one_of_a_pair_rejected():
    a = _image("a.tif", crs="EPSG:32643", size=32)
    b = _image("b.tif", crs=None, size=32)
    result = validate_images([ImageInput(a, "optical"), ImageInput(b, "optical")])
    assert result.ok is False
    assert "CRS" in result.reason


# --- co-registration -------------------------------------------------------------


def _pair_with_known_shift(shift_px):
    size = 80
    rng = np.random.default_rng(42)
    base = rng.random((size, size)) * 255
    shifted = ndi_shift(base, shift_px, mode="reflect")

    img_a = _image("a.tif", bands=1, size=size, data=base[None, :, :].astype("uint8"))
    img_b = _image("b.tif", bands=1, size=size, data=shifted[None, :, :].astype("uint8"))
    return img_a, img_b, base, shifted


def test_small_offset_is_auto_shifted():
    img_a, img_b, base, _shifted = _pair_with_known_shift((2.0, -1.0))
    result = validate_images([ImageInput(img_a, "optical"), ImageInput(img_b, "optical")])

    assert result.ok is True
    assert result.auto_shifted is True
    assert result.fallback_single_modality is False
    assert result.shift_px is not None
    assert result.shifted_images is not None

    # the correction should bring image b close back to image a
    corrected = result.shifted_images[1].data[0].astype(float)
    mse_before = np.mean((img_b.data[0].astype(float) - base) ** 2)
    mse_after = np.mean((corrected - base) ** 2)
    assert mse_after < mse_before


def test_large_offset_flags_and_falls_back_not_hard_reject():
    img_a, img_b, _base, _shifted = _pair_with_known_shift((25.0, 10.0))
    result = validate_images(
        [ImageInput(img_a, "optical"), ImageInput(img_b, "optical")], max_shift_px=5.0
    )

    assert result.ok is True  # NOT a hard rejection
    assert result.fallback_single_modality is True
    assert result.auto_shifted is False
    assert result.shifted_images is None
    assert "misaligned" in result.reason


def test_identical_images_are_aligned_with_zero_shift():
    size = 64
    data = (np.random.default_rng(7).random((1, size, size)) * 255).astype("uint8")
    img_a = _image("a.tif", bands=1, size=size, data=data)
    img_b = _image("b.tif", bands=1, size=size, data=data.copy())

    result = validate_images([ImageInput(img_a, "optical"), ImageInput(img_b, "optical")])

    assert result.ok is True
    assert result.fallback_single_modality is False
    assert result.shift_px == pytest.approx((0.0, 0.0), abs=1e-6)
    assert result.auto_shifted is False  # nothing to correct


def test_different_pixel_dimensions_falls_back():
    img_a = _image("a.tif", bands=1, size=64)
    img_b = _image("b.tif", bands=1, size=32)
    result = validate_images([ImageInput(img_a, "optical"), ImageInput(img_b, "optical")])

    assert result.ok is True
    assert result.fallback_single_modality is True
    assert "different pixel dimensions" in result.reason


# --- cross_sensor=True: co-registration by CRS, phase correlation skipped ------
# Optical and SAR measure genuinely different physics, so grayscale phase
# correlation between them isn't a meaningful alignment signal (confirmed
# empirically against real demo_patches pairs -- see raster_io/validate.py's
# own module docstring for the measured skimage error==1.0 finding).


def _matching_pair(*, bands_a=3, bands_b=2, size=32, crs="EPSG:32629", seed=50):
    a = _image("a.tif", bands=bands_a, size=size, crs=crs, seed=seed)
    b = _image("b.tif", bands=bands_b, size=size, crs=crs, seed=seed + 1)
    return a, b


def test_cross_sensor_matching_crs_and_dimensions_passes_with_no_shift_attempted():
    img_a, img_b = _matching_pair()
    result = validate_images(
        [ImageInput(img_a, "optical"), ImageInput(img_b, "sar")],
        require_matching_crs=True, cross_sensor=True,
    )
    assert result.ok is True
    assert result.fallback_single_modality is False
    assert result.shift_px is None       # phase correlation never ran
    assert result.auto_shifted is False
    assert result.shifted_images is None


def test_cross_sensor_never_calls_phase_cross_correlation(monkeypatch):
    # The real, direct proof this is a genuine skip, not just a code path
    # that happens to also return no shift: phase_cross_correlation itself
    # must never be invoked when cross_sensor=True.
    def _explode(*args, **kwargs):
        raise AssertionError("phase_cross_correlation must not be called for a cross_sensor pair")

    monkeypatch.setattr(validate_module, "phase_cross_correlation", _explode)

    img_a, img_b = _matching_pair()
    result = validate_images(
        [ImageInput(img_a, "optical"), ImageInput(img_b, "sar")],
        require_matching_crs=True, cross_sensor=True,
    )
    assert result.ok is True  # did not raise -- phase_cross_correlation was genuinely skipped


def test_cross_sensor_records_the_skip_in_the_trace():
    img_a, img_b = _matching_pair()
    result = validate_images(
        [ImageInput(img_a, "optical"), ImageInput(img_b, "sar")],
        require_matching_crs=True, cross_sensor=True,
    )
    coreg_entries = [e for e in result.trace if e.task == "validate_images:coregistration"]
    assert len(coreg_entries) == 1
    assert coreg_entries[0].parameters == {"cross_sensor": True}
    assert coreg_entries[0].output["outcome"] == "shift_estimation_skipped"
    assert "not meaningful" in coreg_entries[0].output["reason"]


def test_cross_sensor_mismatched_dimensions_falls_back_not_hard_reject():
    img_a = _image("a.tif", bands=3, size=64, crs="EPSG:32629")
    img_b = _image("b.tif", bands=2, size=32, crs="EPSG:32629")
    result = validate_images(
        [ImageInput(img_a, "optical"), ImageInput(img_b, "sar")],
        require_matching_crs=True, cross_sensor=True,
    )
    assert result.ok is True  # not a hard rejection
    assert result.fallback_single_modality is True
    assert "different pixel dimensions" in result.reason
    assert result.shift_px is None


def test_cross_sensor_with_require_matching_crs_true_rejects_missing_crs():
    # No CRS on either side, and no phase-correlation fallback available
    # anymore for a cross-sensor pair -- genuinely nothing left to verify
    # co-registration by, so this is a hard rejection.
    img_a = _image("a.tif", bands=3, size=32, crs=None)
    img_b = _image("b.tif", bands=2, size=32, crs=None)
    result = validate_images(
        [ImageInput(img_a, "optical"), ImageInput(img_b, "sar")],
        require_matching_crs=True, cross_sensor=True,
    )
    assert result.ok is False
    assert "CRS" in result.reason


def test_cross_sensor_with_mismatched_crs_is_rejected():
    img_a = _image("a.tif", bands=3, size=32, crs="EPSG:32629")
    img_b = _image("b.tif", bands=2, size=32, crs="EPSG:32633")
    result = validate_images(
        [ImageInput(img_a, "optical"), ImageInput(img_b, "sar")],
        require_matching_crs=True, cross_sensor=True,
    )
    assert result.ok is False
    assert "different CRS" in result.reason


# --- trace ------------------------------------------------------------------------


def test_every_call_produces_a_trace():
    result = validate_images([ImageInput(_image(), "optical")])
    assert len(result.trace) > 0
    for entry in result.trace:
        assert entry.tool == "raster_io.validate.validate_images"
        assert entry.confidence == 1.0


def test_rejection_trace_records_the_reason():
    result = validate_images([ImageInput(_image(gsd_metres=None), "optical")])
    assert result.trace[-1].output["reason"] == result.reason
