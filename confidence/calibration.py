"""Temperature scaling: the standard post-hoc fix (Guo et al., 2017, "On
Calibration of Modern Neural Networks") for a neural network's raw softmax
being systematically overconfident -- exactly the failure mode CLAUDE.md's
confidence work exists to correct: "raw CNN softmax is overconfident and
must never be reported as a percentage."

Temperature scaling in probability space, without touching perception/infer.py
--------------------------------------------------------------------------------
Standard temperature scaling divides the model's raw LOGITS by a fitted
scalar T before the softmax: softmax(z / T). perception/infer.py's
InferenceResult only carries the POST-softmax probabilities (`probabilities`),
not the raw logits -- by design, so nothing downstream of it ever has to
re-derive a softmax correctly. Re-deriving temperature scaling from
probabilities alone is still exact, not an approximation, because of a
plain algebraic identity:

    softmax(z / T)_i = exp(z_i / T) / sum_j exp(z_j / T)
                      = (exp(z_i))^(1/T) / sum_j (exp(z_j))^(1/T)      [a^(1/T) := exp(a/T)]
                      = (p_i * Z)^(1/T) / sum_j (p_j * Z)^(1/T)         [p = softmax(z), Z = sum_j exp(z_j)]
                      = p_i^(1/T) / sum_j p_j^(1/T)                    [the Z^(1/T) factor cancels]

where p = softmax(z) is exactly what perception/infer.py already returns.
So `temperature_scale(p, T)` below -- raise every probability to the power
1/T, then renormalize -- is bit-for-bit the same operation as scaling
logits by 1/T and re-applying softmax, just computed from the softmax
output instead of from logits that were never kept around.

T > 1 flattens the distribution (lower peak probability, narrower gap
between the top classes -- less overconfident); T < 1 sharpens it; T = 1
is a no-op. A well-calibrated T is always > 1 for a network that's
overconfident, which is the near-universal finding for modern CNNs
(including this one -- see fit_temperature_from_patches's own printed
diagnostics when it's run).

Fitting T
--------------
fit_temperature() finds the T minimizing the negative log-likelihood (NLL)
of the TRUE class's calibrated probability, over a labelled sample --
exactly Guo et al.'s own objective, and equivalent to minimizing
KL-divergence between the calibrated distribution and the one-hot true
label. A single scalar T can only rescale confidence, never change which
class the argmax picks -- so calibration never changes the model's
predicted MASK, only how confidently that mask's per-pixel probabilities
should be reported. This is why temperature scaling, unlike other
calibration methods, is safe to apply after the fact without touching
evidence/ops.py's downstream computation of the mask at all.

fit_temperature_from_patches() runs this against the real, per-pixel
ground-truth masks in data/bench_patches/*.npz -- the largest set of real
Sentinel-1+2 stacks with genuine annotated masks available locally (64
patches; see eval/run_benchmark.py's own docstring for why this, not the
much larger reBEN corpus, is what's stored locally at all -- CLAUDE.md /
data/README.md). This is NOT the original model's own held-out training/
validation split (that split lives only inside the Kaggle training
notebook per CLAUDE.md, never downloaded locally) -- it is the best real,
disclosed proxy for one that this codebase actually has on disk, the same
proxy eval/run_benchmark.py already treats as its own "real ground truth."
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CONFIDENCE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CONFIDENCE_DIR.parent
DEFAULT_TEMPERATURE_PATH = CONFIDENCE_DIR / "temperature.json"
DEFAULT_PATCHES_DIR = ROOT_DIR / "data" / "bench_patches"

# T=1.0 is the calibration no-op -- used whenever no fitted value has been
# saved yet (confidence/temperature.json doesn't exist), so every function
# in confidence/engine.py that depends on calibration degrades to "trust
# the model's raw softmax" rather than crashing when scripts/fit_calibration.py
# hasn't been run yet.
IDENTITY_TEMPERATURE = 1.0


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Rescales a softmax probability array (any shape, as long as `axis`
    sums to 1) by temperature `T` -- see this module's docstring for why
    `p^(1/T)`, renormalized, is exactly standard logit-space temperature
    scaling. `axis=0` matches perception.infer.InferenceResult.probabilities'
    own (nclasses, H, W) convention."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if temperature == IDENTITY_TEMPERATURE:
        return probabilities
    # Clip away from exactly 0 before the power -- 0^(1/T) is fine (stays
    # 0), but floating-point probabilities that underflowed to a tiny
    # negative value from prior arithmetic would raise under a fractional
    # power; real softmax output is never negative, this is just a safety
    # clamp against that edge case.
    scaled = np.clip(probabilities, 0.0, 1.0) ** (1.0 / temperature)
    return scaled / np.sum(scaled, axis=0, keepdims=True)


def negative_log_likelihood(probabilities: np.ndarray, true_labels: np.ndarray) -> float:
    """Mean -log(P(true class)) over every pixel -- `probabilities` is
    (nclasses, N) or (nclasses, H, W), `true_labels` the matching (N,) or
    (H, W) integer ground truth. Lower is better calibrated; this is
    exactly Guo et al.'s own fitting objective."""
    nclasses = probabilities.shape[0]
    flat_probs = probabilities.reshape(nclasses, -1)
    flat_labels = true_labels.reshape(-1)
    true_class_probs = flat_probs[flat_labels, np.arange(flat_labels.size)]
    # 1e-12 floor: a true class the (mis-)calibrated model assigned
    # probability exactly 0 must not make the whole objective -inf and
    # derail the search -- a heavily but finitely penalized near-zero
    # probability is the honest, numerically stable choice.
    return float(-np.mean(np.log(np.clip(true_class_probs, 1e-12, 1.0))))


def expected_calibration_error(probabilities: np.ndarray, true_labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE: bins predictions by their own top-1 confidence, and averages
    |confidence - accuracy| within each bin, weighted by bin size -- the
    standard, human-readable calibration diagnostic (Naeini et al., 2015):
    "when this model says 80% confident, is it actually right about 80%
    of the time?" reduced to one number. Reported alongside NLL (the
    actual fitting objective) purely as an interpretable sanity check --
    fit_temperature() itself minimizes NLL, not this."""
    nclasses = probabilities.shape[0]
    flat_probs = probabilities.reshape(nclasses, -1)
    flat_labels = true_labels.reshape(-1)

    top1_confidence = np.max(flat_probs, axis=0)
    top1_prediction = np.argmax(flat_probs, axis=0)
    correct = (top1_prediction == flat_labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = flat_labels.size
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (top1_confidence > lo) & (top1_confidence <= hi) if lo > 0 else (
            (top1_confidence >= lo) & (top1_confidence <= hi)
        )
        count = int(np.sum(in_bin))
        if count == 0:
            continue
        bin_confidence = float(np.mean(top1_confidence[in_bin]))
        bin_accuracy = float(np.mean(correct[in_bin]))
        ece += (count / total) * abs(bin_confidence - bin_accuracy)
    return ece


def fit_temperature(
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    *,
    search_range: tuple[float, float] = (0.25, 8.0),
    n_steps: int = 200,
) -> float:
    """The T in `search_range` minimizing negative_log_likelihood(
    temperature_scale(probabilities, T), true_labels), found by a plain
    grid search over `n_steps` candidates.

    A grid search, not a gradient-based optimizer, deliberately: this is a
    ONE-dimensional, smooth, unimodal-in-practice objective computed once
    (fit_temperature_from_patches below is a one-time, offline step, not
    something run per query), so the simplest correct method that can't
    diverge or get stuck in a local optimum is preferable to pulling in an
    optimizer dependency for one scalar.
    """
    lo, hi = search_range
    candidates = np.linspace(lo, hi, n_steps)
    losses = [negative_log_likelihood(temperature_scale(probabilities, float(t)), true_labels) for t in candidates]
    best_index = int(np.argmin(losses))
    return float(candidates[best_index])


@dataclass
class CalibrationReport:
    """Everything scripts/fit_calibration.py prints and saves -- the
    fitted temperature plus honest before/after diagnostics, so a fitted
    value is never just a bare number with no evidence behind it."""

    temperature: float
    n_pixels: int
    n_patches: int
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float
    mean_top1_confidence_before: float
    mean_top1_confidence_after: float
    pixel_accuracy: float


def fit_temperature_from_patches(
    patches_dir: Path = DEFAULT_PATCHES_DIR,
    *,
    session: Any = None,
    config: Any = None,
) -> CalibrationReport:
    """Runs the real model over every real, ground-truth-labelled patch in
    `patches_dir` (sensor='fused', the genuine 16-channel input these
    patches carry -- see eval/run_benchmark.py's own use of this exact
    directory), collects every pixel's (predicted probability vector, true
    class) pair, and fits T against the whole pool at once -- one scalar
    correction for however overconfident this specific model's softmax
    is, not a per-patch or per-class fit (a single global T is Guo et
    al.'s own finding: it already captures nearly all of the achievable
    calibration improvement, and a single number is what
    confidence/engine.py can cheaply apply to any future scene without
    needing this patch set again).
    """
    from perception import infer

    session = session or infer.load_model()
    config = config or infer.load_class_config()

    patch_paths = sorted(Path(patches_dir).glob("*.npz"))
    if not patch_paths:
        raise FileNotFoundError(f"no .npz patches found under {patches_dir}")

    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for path in patch_paths:
        data = np.load(path, allow_pickle=True)
        stack, true_mask = data["stack"][:16].astype(np.float32), data["mask"]
        result = infer.segment_image(stack, "fused", session=session, config=config)
        all_probs.append(result.probabilities)
        all_labels.append(true_mask)

    nclasses = all_probs[0].shape[0]
    probabilities = np.concatenate([p.reshape(nclasses, -1) for p in all_probs], axis=1)
    true_labels = np.concatenate([m.reshape(-1) for m in all_labels])

    nll_before = negative_log_likelihood(probabilities, true_labels)
    ece_before = expected_calibration_error(probabilities, true_labels)
    mean_conf_before = float(np.mean(np.max(probabilities, axis=0)))

    temperature = fit_temperature(probabilities, true_labels)

    calibrated = temperature_scale(probabilities, temperature)
    nll_after = negative_log_likelihood(calibrated, true_labels)
    ece_after = expected_calibration_error(calibrated, true_labels)
    mean_conf_after = float(np.mean(np.max(calibrated, axis=0)))
    pixel_accuracy = float(np.mean(np.argmax(probabilities, axis=0) == true_labels))

    return CalibrationReport(
        temperature=temperature,
        n_pixels=int(true_labels.size),
        n_patches=len(patch_paths),
        nll_before=nll_before,
        nll_after=nll_after,
        ece_before=ece_before,
        ece_after=ece_after,
        mean_top1_confidence_before=mean_conf_before,
        mean_top1_confidence_after=mean_conf_after,
        pixel_accuracy=pixel_accuracy,
    )


def save_temperature(report: CalibrationReport, path: Path = DEFAULT_TEMPERATURE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "temperature": report.temperature,
        "n_pixels": report.n_pixels,
        "n_patches": report.n_patches,
        "nll_before": report.nll_before,
        "nll_after": report.nll_after,
        "ece_before": report.ece_before,
        "ece_after": report.ece_after,
        "mean_top1_confidence_before": report.mean_top1_confidence_before,
        "mean_top1_confidence_after": report.mean_top1_confidence_after,
        "pixel_accuracy": report.pixel_accuracy,
    }, indent=2), encoding="utf-8")


def load_temperature(path: Path = DEFAULT_TEMPERATURE_PATH) -> float:
    """The fitted temperature, or IDENTITY_TEMPERATURE (1.0, a no-op) if
    scripts/fit_calibration.py hasn't been run yet -- confidence/engine.py
    must never crash or refuse to answer just because this one-time
    offline step wasn't performed; it should just fall back to reporting
    the model's raw (still disclosed as such) softmax margin."""
    if not path.exists():
        return IDENTITY_TEMPERATURE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        temperature = float(data["temperature"])
        return temperature if temperature > 0 else IDENTITY_TEMPERATURE
    except Exception:
        return IDENTITY_TEMPERATURE
