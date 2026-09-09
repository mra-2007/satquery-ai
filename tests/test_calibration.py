"""Tests for confidence/calibration.py: temperature scaling's mathematical
equivalence to logit-space scaling, the NLL/ECE diagnostics, and
fit_temperature()'s ability to recover a known miscalibration."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from confidence.calibration import (
    IDENTITY_TEMPERATURE,
    expected_calibration_error,
    fit_temperature,
    fit_temperature_from_patches,
    load_temperature,
    negative_log_likelihood,
    save_temperature,
    temperature_scale,
)

RNG = np.random.default_rng(0)


def _softmax(logits: np.ndarray, axis: int = 0) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


# --- temperature_scale ---------------------------------------------------------


def test_temperature_scale_is_a_no_op_at_identity():
    probs = _softmax(RNG.normal(size=(5, 4, 4)))
    np.testing.assert_allclose(temperature_scale(probs, IDENTITY_TEMPERATURE), probs)


def test_temperature_scale_rejects_non_positive_temperature():
    probs = _softmax(RNG.normal(size=(3, 2, 2)))
    with pytest.raises(ValueError):
        temperature_scale(probs, 0.0)
    with pytest.raises(ValueError):
        temperature_scale(probs, -1.0)


def test_temperature_scale_still_sums_to_one():
    probs = _softmax(RNG.normal(size=(6, 5, 5)))
    for t in (0.3, 0.7, 1.5, 4.0):
        scaled = temperature_scale(probs, t)
        np.testing.assert_allclose(np.sum(scaled, axis=0), 1.0, atol=1e-10)


def test_temperature_scale_greater_than_one_flattens_the_distribution():
    probs = _softmax(RNG.normal(size=(8, 10, 10)) * 3)  # sharp, confident logits
    peak_before = np.max(probs, axis=0)
    peak_after = np.max(temperature_scale(probs, 2.0), axis=0)
    assert np.all(peak_after < peak_before)


def test_temperature_scale_less_than_one_sharpens_the_distribution():
    probs = _softmax(RNG.normal(size=(8, 10, 10)) * 3)
    peak_before = np.max(probs, axis=0)
    peak_after = np.max(temperature_scale(probs, 0.5), axis=0)
    assert np.all(peak_after > peak_before)


def test_temperature_scale_matches_logit_space_scaling_exactly():
    # The core claim of confidence/calibration.py's own module docstring:
    # p^(1/T) renormalized, computed from softmax(z) alone, is bit-for-bit
    # the same as softmax(z / T) computed directly from the logits.
    logits = RNG.normal(size=(7, 6, 6)) * 2.5
    probs = _softmax(logits)
    for t in (0.4, 1.0, 1.7, 3.3):
        via_probabilities = temperature_scale(probs, t)
        via_logits = _softmax(logits / t)
        np.testing.assert_allclose(via_probabilities, via_logits, atol=1e-9)


# --- negative_log_likelihood ---------------------------------------------------


def test_nll_near_zero_for_confident_correct_predictions():
    nclasses = 4
    labels = np.array([0, 1, 2, 3])
    probs = np.full((nclasses, 4), 1e-6)
    probs[labels, np.arange(4)] = 1.0 - 1e-6 * (nclasses - 1)
    assert negative_log_likelihood(probs, labels) < 1e-4


def test_nll_equals_log_nclasses_for_uniform_predictions():
    nclasses = 5
    labels = RNG.integers(0, nclasses, size=100)
    probs = np.full((nclasses, 100), 1.0 / nclasses)
    assert negative_log_likelihood(probs, labels) == pytest.approx(np.log(nclasses), abs=1e-9)


def test_nll_handles_hw_shaped_labels():
    nclasses = 3
    labels = np.zeros((4, 4), dtype=int)
    probs = np.full((nclasses, 4, 4), 1.0 / nclasses)
    assert negative_log_likelihood(probs, labels) == pytest.approx(np.log(nclasses), abs=1e-9)


# --- expected_calibration_error -------------------------------------------------


def test_ece_is_zero_for_a_perfectly_calibrated_single_bin():
    # 100 predictions, all with top-1 confidence exactly 0.7, and exactly
    # 70 of them correct -- confidence and accuracy match exactly in the
    # one bin they all fall into.
    nclasses = 2
    n = 100
    probs = np.zeros((nclasses, n))
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        correct = i < 70
        labels[i] = 0 if correct else 1
        probs[:, i] = [0.7, 0.3]
    assert expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.0, abs=1e-9)


def test_ece_is_positive_when_confidence_and_accuracy_diverge():
    # Every prediction claims 0.99 confidence but is only right half the time.
    nclasses = 2
    n = 100
    probs = np.tile([[0.99], [0.01]], (1, n))
    labels = np.array([0, 1] * (n // 2))
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert ece == pytest.approx(0.49, abs=0.02)


# --- fit_temperature -------------------------------------------------------------


def test_fit_temperature_recovers_a_known_overconfidence_scale():
    # z are the TRUE, well-calibrated logits: labels are drawn from
    # softmax(z) itself. The "raw model" is simulated as sharpened by a
    # known factor k=3 (raw_probs = softmax(z * k)) -- exactly the
    # overconfident-CNN failure mode this module exists to correct.
    # temperature_scale(raw_probs, T=k) recovers softmax(z) exactly (see
    # confidence/calibration.py's own docstring identity), so fitting
    # against labels drawn from softmax(z) should recover T close to k.
    nclasses, n_samples = 4, 20_000
    k = 3.0

    z = RNG.normal(size=(nclasses, n_samples)) * 1.5
    true_probs = _softmax(z)
    labels = np.array([RNG.choice(nclasses, p=true_probs[:, i]) for i in range(n_samples)])
    raw_probs = _softmax(z * k)

    fitted_t = fit_temperature(raw_probs, labels, search_range=(0.5, 6.0), n_steps=200)
    assert fitted_t == pytest.approx(k, abs=0.5)


def test_fit_temperature_of_already_calibrated_data_is_near_one():
    nclasses, n_samples = 4, 20_000
    z = RNG.normal(size=(nclasses, n_samples)) * 1.5
    true_probs = _softmax(z)
    labels = np.array([RNG.choice(nclasses, p=true_probs[:, i]) for i in range(n_samples)])

    fitted_t = fit_temperature(true_probs, labels, search_range=(0.25, 8.0), n_steps=200)
    assert fitted_t == pytest.approx(1.0, abs=0.5)


# --- fit_temperature_from_patches / save_temperature / load_temperature --------


@pytest.fixture(scope="module")
def two_real_patches(tmp_path_factory) -> Path:
    """A tiny, fast subset (2 patches, not all 64) of the same real,
    ground-truth-labelled data/bench_patches/*.npz eval/run_benchmark.py
    and scripts/fit_calibration.py use -- enough to exercise
    fit_temperature_from_patches() against genuine model output without
    the full 64-patch sweep's runtime in the regular test suite."""
    source_dir = Path(__file__).resolve().parent.parent / "data" / "bench_patches"
    patch_paths = sorted(source_dir.glob("*.npz"))[:2]
    assert len(patch_paths) == 2, "expected at least 2 real patches under data/bench_patches"

    dest_dir = tmp_path_factory.mktemp("calibration_patches")
    for path in patch_paths:
        shutil.copy(path, dest_dir / path.name)
    return dest_dir


def test_fit_temperature_from_patches_returns_a_sane_report(two_real_patches):
    report = fit_temperature_from_patches(two_real_patches)

    assert report.n_patches == 2
    assert report.n_pixels == 2 * 120 * 120
    assert 0.0 <= report.pixel_accuracy <= 1.0
    assert report.nll_before >= 0.0
    assert report.nll_after >= 0.0
    assert report.ece_before >= 0.0
    assert report.ece_after >= 0.0
    assert 0.0 < report.temperature  # within fit_temperature's own default search_range


def test_fit_temperature_from_patches_raises_on_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        fit_temperature_from_patches(tmp_path)


def test_save_and_load_temperature_round_trips(two_real_patches, tmp_path):
    report = fit_temperature_from_patches(two_real_patches)
    path = tmp_path / "temperature.json"

    save_temperature(report, path)
    assert load_temperature(path) == pytest.approx(report.temperature)


def test_load_temperature_defaults_to_identity_when_file_missing(tmp_path):
    assert load_temperature(tmp_path / "does_not_exist.json") == IDENTITY_TEMPERATURE


def test_load_temperature_defaults_to_identity_on_corrupt_file(tmp_path):
    path = tmp_path / "temperature.json"
    path.write_text("not valid json", encoding="utf-8")
    assert load_temperature(path) == IDENTITY_TEMPERATURE


def test_load_temperature_rejects_a_non_positive_saved_value(tmp_path):
    path = tmp_path / "temperature.json"
    path.write_text('{"temperature": -2.0}', encoding="utf-8")
    assert load_temperature(path) == IDENTITY_TEMPERATURE
