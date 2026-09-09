"""Fits confidence/calibration.py's temperature scaling against every real,
ground-truth-labelled patch in data/bench_patches/*.npz, prints honest
before/after diagnostics (NLL, ECE, mean top-1 confidence, pixel accuracy),
and saves the fitted value to confidence/temperature.json --
confidence/engine.py loads that file at runtime (falling back to the
calibration no-op, T=1.0, if this script has never been run).

See confidence/calibration.py's own module docstring for why
data/bench_patches (not a separate held-out validation split, which isn't
stored locally at all -- CLAUDE.md / data/README.md) is the real,
disclosed proxy this project actually has for one.

Run with the venv's own interpreter:
    venv\\Scripts\\python.exe -m scripts.fit_calibration
"""

from confidence.calibration import (
    DEFAULT_PATCHES_DIR,
    DEFAULT_TEMPERATURE_PATH,
    fit_temperature_from_patches,
    save_temperature,
)


def main() -> None:
    print(__doc__)
    print(f"Running the real model over every patch in {DEFAULT_PATCHES_DIR} ...\n")

    report = fit_temperature_from_patches()

    print(f"Patches used:            {report.n_patches}")
    print(f"Pixels used:             {report.n_pixels:,}")
    print(f"Raw pixel accuracy:      {report.pixel_accuracy:.4f}")
    print()
    print(f"{'':20s}{'BEFORE (T=1.0)':>18s}{'AFTER (fitted T)':>18s}")
    print(f"{'NLL':20s}{report.nll_before:>18.4f}{report.nll_after:>18.4f}")
    print(f"{'ECE':20s}{report.ece_before:>18.4f}{report.ece_after:>18.4f}")
    print(f"{'Mean top-1 conf.':20s}{report.mean_top1_confidence_before:>18.4f}{report.mean_top1_confidence_after:>18.4f}")
    print()
    print(f"Fitted temperature: T = {report.temperature:.4f}")
    if report.temperature > 1.0:
        print(
            f"T > 1: the raw softmax IS overconfident, as expected -- mean top-1 confidence "
            f"drops from {report.mean_top1_confidence_before:.1%} to {report.mean_top1_confidence_after:.1%} "
            f"after calibration, much closer to the real {report.pixel_accuracy:.1%} pixel accuracy."
        )
    elif report.temperature < 1.0:
        print("T < 1: the raw softmax was, unusually, UNDERconfident on this sample.")
    else:
        print("T = 1: the raw softmax was already well calibrated on this sample (a no-op fit).")

    save_temperature(report, DEFAULT_TEMPERATURE_PATH)
    print(f"\nSaved to {DEFAULT_TEMPERATURE_PATH}")


if __name__ == "__main__":
    main()
