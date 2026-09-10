"""Validates one or two RasterImages before any model runs.

Checks, in order, with a clear human-readable rejection reason the moment
one fails: image count, format, modality, GSD, band count, and (for a
pair) CRS and pixel co-registration. Per CLAUDE.md's "No pixel, no claim":
an image with no resolvable GSD is refused outright, never silently
defaulted (e.g. to 10 m).

Co-registration, when exactly two images are given, normally uses
skimage.registration.phase_cross_correlation on grayscale versions of both
rasters. A small detected offset (<= max_shift_px) is corrected in place
with scipy.ndimage.shift and the run proceeds normally; a larger one is
NOT a hard rejection -- it's flagged (`fallback_single_modality=True`,
with `reason` explaining why) so the caller can still answer using just
one of the two images instead of aborting entirely.

Cross-sensor pairs (`cross_sensor=True`, api/main.py's cross_modal upload)
skip phase correlation entirely instead: optical and SAR measure genuinely
different physics (reflectance vs. radar backscatter), so a naive
grayscale correlation has no real shared structure to lock onto.
Confirmed empirically, not assumed -- skimage's own degenerate-correlation
signal (error == 1.0, "no meaningful peak found") appears even for
demo_patches pairs known to be pixel-perfect aligned, and the "shift" it
reports in that regime is noise, not a real misalignment measurement; a
real satquery-ai demo upload was wrongly rejected as misaligned by exactly
this before this fix existed. Dimensions and CRS are verified instead
(CRS via `require_matching_crs=True`, the caller's job to pass; dimensions
here) and the skip itself is recorded in the trace, never silently
assumed. Bi-temporal ("change") pairs keep the original phase-correlation
check unchanged -- both images there come from the SAME sensor, where
grayscale correlation is a meaningful alignment signal.
"""

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation

from evidence.schema import TraceStep
from raster_io.readers import SUPPORTED_EXTENSIONS, RasterImage

DEFAULT_MAX_SHIFT_PX = 5.0


@dataclass
class ImageInput:
    """One image plus the modality it was declared as -- modality can't be
    reliably inferred from pixels alone, so the caller states it."""

    image: RasterImage
    modality: str  # "optical" or "sar"


@dataclass
class ValidationResult:
    ok: bool
    reason: str | None
    shift_px: tuple[float, float] | None
    auto_shifted: bool
    fallback_single_modality: bool
    shifted_images: list[RasterImage] | None
    trace: list[TraceStep]


def _trace(check: str, parameters: dict, output: dict, confidence: float) -> TraceStep:
    return TraceStep(
        task=f"validate_images:{check}",
        tool="raster_io.validate.validate_images",
        parameters=parameters,
        output=output,
        confidence=confidence,
    )


def _to_grayscale(data: np.ndarray) -> np.ndarray:
    if data.shape[0] == 1:
        return data[0].astype(float)
    return data.mean(axis=0).astype(float)


def _apply_shift(image: RasterImage, shift_px: tuple[float, float]) -> RasterImage:
    """Correct `image`'s pixel content by `shift_px` (as returned by
    phase_cross_correlation against a reference). Only the pixel data is
    corrected -- the transform/CRS are left as read, since the shift is a
    registration correction, not a change of where the file itself claims
    to be located."""
    shifted = np.stack([
        ndi_shift(band.astype(float), shift_px, mode="nearest")
        for band in image.data
    ])
    return replace(image, data=shifted.astype(image.data.dtype))


def validate_images(
    inputs: list[ImageInput],
    *,
    expected_count: int | None = None,
    expected_modality: str | set[str] | None = None,
    expected_gsd_range: tuple[float, float] | None = None,
    min_band_count: int | None = None,
    max_shift_px: float = DEFAULT_MAX_SHIFT_PX,
    require_matching_crs: bool = True,
    cross_sensor: bool = False,
) -> ValidationResult:
    """Validate `inputs` (1 or 2 ImageInputs) before any model is allowed
    to run on them.

    `ok=False` means: reject with `reason`, run nothing. `ok=True` with
    `fallback_single_modality=True` means: the pair failed co-registration
    (by more than `max_shift_px`, or -- for a `cross_sensor` pair -- a
    dimension mismatch), so proceed with single-image analysis only --
    `reason` still explains why, for the trace/UI, but this is not a hard
    rejection.

    `require_matching_crs` (default True) rejects a pair outright when
    either image lacks CRS metadata or the two disagree -- correct for
    georeferenced sources, where mismatched CRS makes any pixel
    comparison meaningless. Pass False when the caller is instead relying
    on phase_cross_correlation itself as the alignment check (e.g. two
    plain PNG/JPEG uploads the caller already knows cover the same
    footprint) -- CRS is then skipped, not assumed to match.

    `cross_sensor` (default False) is for a declared optical+SAR pair
    (api/main.py's cross_modal upload): phase correlation between two
    genuinely different sensing modalities is not a meaningful alignment
    signal (see this module's own docstring for why -- confirmed
    empirically, not assumed), so it's skipped entirely rather than run
    and trusted anyway. Pass `require_matching_crs=True` alongside it --
    CRS becomes the actual verification co-registration relies on here,
    not an optional extra -- and only a matching dimensions/CRS pair is
    treated as aligned; the skip itself is always recorded in the trace.
    Leave False (the default) for a same-sensor ("change") pair, where
    grayscale phase correlation remains a real, meaningful check.
    """
    trace: list[TraceStep] = []

    def _reject(reason: str) -> ValidationResult:
        trace.append(_trace("reject", {}, {"ok": False, "reason": reason}, 1.0))
        return ValidationResult(
            ok=False, reason=reason, shift_px=None, auto_shifted=False,
            fallback_single_modality=False, shifted_images=None, trace=trace,
        )

    # 1. image count
    if expected_count is not None and len(inputs) != expected_count:
        return _reject(f"expected {expected_count} image(s), got {len(inputs)}")
    if len(inputs) == 0:
        return _reject("no images provided")
    if len(inputs) > 2:
        return _reject(f"validate_images supports at most 2 images (a pair), got {len(inputs)}")

    allowed_modalities = None
    if expected_modality is not None:
        allowed_modalities = {expected_modality} if isinstance(expected_modality, str) else set(expected_modality)

    for i, item in enumerate(inputs):
        img = item.image

        # 2. format (belt-and-suspenders: read_image() already enforces this)
        if img.path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return _reject(f"image {i} ({img.path.name}): unsupported format {img.path.suffix!r}")

        # 3. modality
        if allowed_modalities is not None and item.modality not in allowed_modalities:
            return _reject(
                f"image {i} ({img.path.name}): modality {item.modality!r} "
                f"not in {sorted(allowed_modalities)}"
            )

        # 4. GSD -- refuse rather than default
        if img.gsd_metres is None:
            return _reject(
                f"image {i} ({img.path.name}): no resolvable GSD -- refusing rather than "
                "assuming a default resolution"
            )
        if expected_gsd_range is not None:
            lo, hi = expected_gsd_range
            if not (lo <= img.gsd_metres <= hi):
                return _reject(
                    f"image {i} ({img.path.name}): GSD {img.gsd_metres:.3f} m outside "
                    f"expected range {expected_gsd_range}"
                )

        # 5. band count
        if min_band_count is not None and img.band_count < min_band_count:
            return _reject(
                f"image {i} ({img.path.name}): has {img.band_count} band(s), needs "
                f">= {min_band_count}"
            )

    trace.append(_trace("per_image", {"count": len(inputs)}, {"ok": True}, 1.0))

    # 6. CRS -- must match across a pair, unless the caller opted out
    if len(inputs) == 2:
        if not require_matching_crs:
            trace.append(_trace(
                "crs", {"require_matching_crs": False},
                {"ok": True, "outcome": "skipped -- relying on phase correlation instead"}, 1.0,
            ))
        else:
            crs_values = {item.image.crs for item in inputs}
            if None in crs_values:
                return _reject("cannot compare a pair of images without CRS metadata on both of them")
            if len(crs_values) > 1:
                return _reject(f"images use different CRS: {sorted(crs_values)}")
            trace.append(_trace("crs", {"crs": next(iter(crs_values))}, {"ok": True}, 1.0))

    # 7. co-registration -- only meaningful for a pair
    shift_px = None
    auto_shifted = False
    fallback_single_modality = False
    shifted_images = None
    reason = None

    if len(inputs) == 2:
        img_a, img_b = inputs[0].image, inputs[1].image

        if cross_sensor:
            # Phase correlation is skipped entirely -- see this module's
            # own docstring and validate_images()'s `cross_sensor`
            # parameter for why it isn't a meaningful signal between
            # optical and SAR. Only dimensions are checked here; CRS was
            # already verified above (require_matching_crs=True is the
            # caller's job to pass alongside cross_sensor=True) -- between
            # them, that IS the co-registration verification for this
            # pair, disclosed as such in the trace rather than silently
            # assumed or silently skipped.
            shape_a, shape_b = img_a.data.shape[1:], img_b.data.shape[1:]
            if shape_a != shape_b:
                fallback_single_modality = True
                reason = (
                    f"images have different pixel dimensions ({shape_a} vs {shape_b}); "
                    "falling back to single-modality analysis"
                )
                trace.append(_trace(
                    "coregistration", {"cross_sensor": True, "shape_a": shape_a, "shape_b": shape_b},
                    {"ok": True, "outcome": "shape_mismatch_fallback"}, 1.0,
                ))
            else:
                trace.append(_trace(
                    "coregistration", {"cross_sensor": True},
                    {
                        "ok": True,
                        "outcome": "shift_estimation_skipped",
                        "reason": "phase correlation between optical and SAR is not meaningful "
                                  "(different sensing physics) -- verified by matching dimensions and CRS instead",
                    },
                    1.0,
                ))

        else:
            gray_a, gray_b = _to_grayscale(img_a.data), _to_grayscale(img_b.data)

            if gray_a.shape != gray_b.shape:
                fallback_single_modality = True
                reason = (
                    f"images have different pixel dimensions ({gray_a.shape} vs {gray_b.shape}); "
                    "falling back to single-modality analysis"
                )
                trace.append(_trace(
                    "coregistration", {"shape_a": gray_a.shape, "shape_b": gray_b.shape},
                    {"ok": True, "outcome": "shape_mismatch_fallback"}, 1.0,
                ))
            else:
                shift, _error, _phasediff = phase_cross_correlation(gray_a, gray_b, upsample_factor=10)
                shift_px = (float(shift[0]), float(shift[1]))
                magnitude = math.hypot(*shift_px)

                if magnitude <= max_shift_px:
                    if magnitude > 0:
                        shifted_images = [img_a, _apply_shift(img_b, shift_px)]
                        auto_shifted = True
                    trace.append(_trace(
                        "coregistration", {"max_shift_px": max_shift_px},
                        {"ok": True, "outcome": "aligned", "shift_px": shift_px, "magnitude_px": magnitude},
                        1.0,
                    ))
                else:
                    fallback_single_modality = True
                    reason = (
                        f"images are misaligned by {magnitude:.2f} px (> {max_shift_px} px threshold); "
                        "falling back to single-modality analysis"
                    )
                    trace.append(_trace(
                        "coregistration", {"max_shift_px": max_shift_px},
                        {"ok": True, "outcome": "misaligned_fallback", "shift_px": shift_px, "magnitude_px": magnitude},
                        1.0,
                    ))

    return ValidationResult(
        ok=True,
        reason=reason,
        shift_px=shift_px,
        auto_shifted=auto_shifted,
        fallback_single_modality=fallback_single_modality,
        shifted_images=shifted_images,
        trace=trace,
    )
