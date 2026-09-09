"""Independent, multi-source evidence fusion for one class's presence in a
scene -- four sources, each asked the same question ("is class_id here?")
by a genuinely different method, so that agreement between them means
something:

1. **Segmentation head** -- perception/infer.py's per-pixel argmax mask,
   sensor_id="fused" (every real channel the stack has).
2. **Classification head** -- the SAME model's separate scene-level
   multi-label tagging head, sigmoid-activated (see perception/infer.py's
   InferenceResult.classification docstring for why sigmoid, not softmax).
3. **Spectral index** -- NDWI (water), NDBI (built-up), or NDVI
   (vegetation), computed directly from the stack's real optical
   reflectance bands via fixed, published formulas. No learned weights at
   all: pure arithmetic on real pixels.
4. **SAR branch** -- perception/infer.py's segmentation head again, but
   sensor_id="sar", read only when the stack actually carries real SAR
   backscatter (see `has_real_sar`) rather than the zero-filled channels
   an optical-only upload gets.

Per CLAUDE.md's "No pixel, no claim": nothing here is asked of or smoothed
by an LLM -- every source is a deterministic read of a real raster, and
`fuse_evidence`'s only job is to combine four already-computed numbers.

**Why sources 1 and 2 are NOT treated as two independent checks.** The
segmentation and classification heads are two output projections on top of
ONE shared encoder, run in the SAME forward pass over the SAME input --
they see identical internal activations and differ only in their final
layer. If the encoder embeds this scene wrong (bad lighting, an unusual
land-cover mix, anything outside its training distribution), BOTH heads
inherit that same wrong embedding and are liable to agree with each other
for the wrong reason. Two heads agreeing is corroborating, not two
independent confirmations -- a lesson this project already learned the
hard way once (see the segmentation-vs-classification label-order mixup
this module is careful never to repeat: `classification` is indexed by
ClassConfig's own alphabetical `labels`, `mask`/`probabilities` by
agent.vocabulary.SEGMENTATION_CLASSES -- never conflate the two orders).

The spectral index (source 3) and the SAR branch (source 4) are each
genuinely independent of that shared encoder: the spectral index touches
no model weights at all, and the SAR branch's INPUT is a physically
different sensing modality (radar backscatter, unaffected by cloud cover
or illumination, versus optical reflectance) even though it happens to
flow through the same trained network. `SOURCE_WEIGHTS` below weights both
of them higher than either head-derived source specifically because of
this -- see `WEIGHTING_NOTE`, which is also carried into every
`FusionResult` so a caller never has to already know this to interpret one.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agent.vocabulary import SEGMENTATION_CLASSES
from evidence import ops
from perception import infer

ClassId = int | list[int]

# --- Real Sentinel-2 optical band channel indices within the model's
# 16-channel stack (0-9 optical, 10-11 SAR VV/VH, 12-15 auxiliary -- see
# tools/cross_modal.py's own docstring for that channel-range boundary).
# This is BigEarthNet's standard 10-band Sentinel-2 selection, ascending
# wavelength order. Channel 0 = B02 and channel 9 = B12 are independently
# documented elsewhere in this codebase (api/scenes.py's render_preview_png,
# tools/cross_modal.py); the six bands between them are cross-checked here
# against models/class_config.json's own per-channel `mean`, which traces
# out exactly the textbook vegetation reflectance curve -- rising
# Blue -> Green -> Red -> three Red-Edge bands -> a NIR plateau, then
# dropping through SWIR1 -> SWIR2 -- ONLY under this ordering.
BLUE, GREEN, RED = 0, 1, 2
RED_EDGE_1, RED_EDGE_2, RED_EDGE_3 = 3, 4, 5
NIR, NARROW_NIR = 6, 7
SWIR1, SWIR2 = 8, 9
SAR_VV, SAR_VH = 10, 11

_NDVI = "ndvi"
_NDWI = "ndwi"
_NDBI = "ndbi"

# The segmentation-order (agent.vocabulary.SEGMENTATION_CLASSES) class name
# each real spectral index is a meaningful presence test for. A class with
# no entry here (Moors/heathland, Beaches/dunes/sands, Inland/Coastal
# wetlands -- none of which has one clean, standard normalized-difference
# formula) reports spectral evidence as unavailable rather than guessing.
SPECTRAL_INDEX_BY_CLASS_NAME: dict[str, str] = {
    "Inland waters": _NDWI,
    "Marine waters": _NDWI,
    "Urban fabric": _NDBI,
    "Industrial or commercial units": _NDBI,
    "Arable land": _NDVI,
    "Permanent crops": _NDVI,
    "Pastures": _NDVI,
    "Complex cultivation patterns": _NDVI,
    "Land principally occupied by agriculture, with significant areas of natural vegetation": _NDVI,
    "Agro-forestry areas": _NDVI,
    "Broad-leaved forest": _NDVI,
    "Coniferous forest": _NDVI,
    "Mixed forest": _NDVI,
    "Natural grassland and sparsely vegetated areas": _NDVI,
    "Transitional woodland, shrub": _NDVI,
}

# Standard, published decision thresholds -- McFeeters (1996) for NDWI > 0,
# Zha et al. (2003) for NDBI > 0; NDVI's 0.3 is the commonly cited cutoff
# separating vegetated from bare/built land in Sentinel-2-resolution
# studies (values below are typically bare soil, water, or built surface).
INDEX_THRESHOLDS: dict[str, float] = {_NDWI: 0.0, _NDBI: 0.0, _NDVI: 0.3}

# Segmentation and classification are correlated (shared encoder, same
# forward pass -- see the module docstring) and therefore weighted equally
# and LOWER; the spectral index and SAR branch are each independent of that
# shared encoder and of each other, and weighted equally and HIGHER.
SOURCE_WEIGHTS: dict[str, float] = {
    "segmentation": 1.0,
    "classification": 1.0,
    "spectral_index": 2.0,
    "sar": 2.0,
}
CORRELATED_SOURCES: tuple[str, ...] = ("segmentation", "classification")

WEIGHTING_NOTE: str = (
    "The segmentation and classification heads share one encoder and run in the same "
    "forward pass, so their errors are correlated -- agreement between them is "
    "corroborating, not two independent checks. The spectral index (raw-band "
    "arithmetic, no learned weights) and the SAR branch (a physically different "
    "sensing modality, real radar backscatter rather than optical reflectance) are "
    "each genuinely independent of that shared encoder and of each other, so both are "
    "weighted higher than head-to-head agreement in this fusion."
)


@dataclass
class SourceEvidence:
    """One source's independent read on whether a class is present.
    `available=False` (with `present`/`confidence` both None) means this
    source had nothing to say for this class/scene -- e.g. no real SAR was
    ever captured, or no spectral index applies to this class -- which is
    disclosed, never silently treated as "absent"."""

    source: str
    available: bool
    present: bool | None
    confidence: float | None  # this source's own [0, 1] confidence when available
    weight: float
    detail: str


@dataclass
class FusionResult:
    class_id: ClassId
    class_name: str
    sources: list[SourceEvidence] = field(default_factory=list)
    # source name -> that source's own present/absent verdict (None if
    # that source was unavailable) -- side-by-side per-source agreement.
    agreement: dict[str, bool | None] = field(default_factory=dict)
    agreement_fraction: float = 0.0  # fraction of AVAILABLE sources that agree with fused_present
    fused_present: bool = False
    fused_confidence: float = 0.0
    correlated_sources: tuple[str, ...] = CORRELATED_SOURCES
    weighting_note: str = WEIGHTING_NOTE
    summary: str = ""


def has_real_sar(stack: np.ndarray) -> bool:
    """True iff the stack's two SAR channels (10=VV, 11=VH) carry real,
    non-zero backscatter, not the all-zero placeholder an optical-only
    upload's stack gets there (see api/scenes.py's build_model_input_from_bands
    -- it only ever fills channels 10-11 from a real second SAR image).
    Real dB backscatter is essentially never exactly 0.0 across every
    single pixel, so an all-zero channel reliably means "no real SAR was
    ever placed here", not an unusually flat real scene."""
    return bool(np.any(stack[SAR_VV] != 0.0) and np.any(stack[SAR_VH] != 0.0))


def _normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    denom = a64 + b64
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom != 0, (a64 - b64) / denom, 0.0)


def ndvi(stack: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index (Rouse et al., 1974): healthy
    vegetation reflects strongly in the NIR and absorbs Red."""
    return _normalized_difference(stack[NIR], stack[RED])


def ndwi(stack: np.ndarray) -> np.ndarray:
    """Normalized Difference Water Index (McFeeters, 1996): open water
    reflects Green more than NIR, which it absorbs almost completely."""
    return _normalized_difference(stack[GREEN], stack[NIR])


def ndbi(stack: np.ndarray) -> np.ndarray:
    """Normalized Difference Built-up Index (Zha et al., 2003): built
    surfaces reflect more SWIR than NIR, unlike vegetation or bare soil."""
    return _normalized_difference(stack[SWIR1], stack[NIR])


_INDEX_FUNCTIONS = {_NDVI: ndvi, _NDWI: ndwi, _NDBI: ndbi}


def _as_list(class_id: ClassId) -> list[int]:
    return class_id if isinstance(class_id, list) else [class_id]


def _class_label(class_id: ClassId) -> str:
    return " / ".join(SEGMENTATION_CLASSES[i] for i in _as_list(class_id))


def _segmentation_evidence(result: infer.InferenceResult, class_id: ClassId, metadata: dict) -> SourceEvidence:
    present = ops.presence(result.mask, class_id, 0.0, metadata)
    # The PEAK softmax probability this class reached anywhere in the
    # scene, not the mean across every pixel. `present` is an existence
    # question (does even one connected component exist?), so confidence
    # has to track the strength of that same claim at its strongest point
    # -- the scene-wide mean would dilute a real but small water body
    # under a lake of "definitely not water" pixels elsewhere in the same
    # frame, understating confidence in something the mask genuinely found.
    confidence = float(np.max(result.probabilities[_as_list(class_id), :, :]))
    return SourceEvidence(
        source="segmentation", available=True, present=present, confidence=confidence,
        weight=SOURCE_WEIGHTS["segmentation"],
        detail="Per-pixel argmax mask, sensor='fused' -- peak class probability anywhere in the scene.",
    )


def _classification_evidence(result: infer.InferenceResult, class_id: ClassId, config: infer.ClassConfig) -> SourceEvidence:
    try:
        indices = [config.labels.index(SEGMENTATION_CLASSES[i]) for i in _as_list(class_id)]
    except ValueError as exc:
        raise ValueError(
            f"class name not found in this model's classification-head vocabulary: {exc}"
        ) from exc
    confidence = float(np.max(result.classification[indices]))
    return SourceEvidence(
        source="classification", available=True, present=confidence >= 0.5, confidence=confidence,
        weight=SOURCE_WEIGHTS["classification"],
        detail="Scene-level multi-label tagging head, sigmoid probability (>=0.5 counts as present).",
    )


def _spectral_evidence(stack: np.ndarray, class_id: ClassId, metadata: dict) -> SourceEvidence:
    names = [SEGMENTATION_CLASSES[i] for i in _as_list(class_id)]
    categories = {SPECTRAL_INDEX_BY_CLASS_NAME.get(name) for name in names}
    if None in categories or len(categories) != 1:
        return SourceEvidence(
            source="spectral_index", available=False, present=None, confidence=None,
            weight=SOURCE_WEIGHTS["spectral_index"],
            detail=f"No single standard spectral index applies to {' / '.join(names)}.",
        )
    category = categories.pop()
    index_array = _INDEX_FUNCTIONS[category](stack)
    threshold = INDEX_THRESHOLDS[category]
    fraction_above = float(np.mean(index_array > threshold))
    pixel_area_m2 = metadata["gsd_metres"] ** 2
    area_m2 = fraction_above * index_array.size * pixel_area_m2
    present = area_m2 > 0.0
    # Peak index value, rescaled from its natural [-1, 1] range to [0, 1]
    # -- same "strongest pixel, not scene-wide average" reasoning as the
    # segmentation/SAR sources: `fraction_above` alone would understate
    # confidence for a small-but-real feature (a real pond covering 2% of
    # the scene is still real, not 2%-confidently real) just because most
    # of the SAME scene is, correctly, something else entirely.
    peak_index = float(np.max(index_array))
    confidence = max(0.0, min(1.0, (peak_index + 1.0) / 2.0))
    return SourceEvidence(
        source="spectral_index", available=True, present=present, confidence=confidence,
        weight=SOURCE_WEIGHTS["spectral_index"],
        detail=f"{category.upper()} > {threshold}: peak {peak_index:.2f} over {fraction_above * 100:.1f}% of the scene.",
    )


def _sar_evidence(
    stack: np.ndarray, class_id: ClassId, metadata: dict, *, session: Any, config: infer.ClassConfig,
) -> SourceEvidence:
    if not has_real_sar(stack):
        return SourceEvidence(
            source="sar", available=False, present=None, confidence=None,
            weight=SOURCE_WEIGHTS["sar"],
            detail="This stack carries no real SAR backscatter (channels 10-11 are zero-filled).",
        )
    sar_result = infer.segment_image(stack, "sar", session=session, config=config)
    present = ops.presence(sar_result.mask, class_id, 0.0, metadata)
    # Peak probability, same reasoning as _segmentation_evidence above.
    confidence = float(np.max(sar_result.probabilities[_as_list(class_id), :, :]))
    return SourceEvidence(
        source="sar", available=True, present=present, confidence=confidence,
        weight=SOURCE_WEIGHTS["sar"],
        detail="Segmentation head, sensor='sar' -- peak class probability anywhere in the scene, real radar backscatter only.",
    )


def _render_summary(class_name: str, sources: list[SourceEvidence], fused_present: bool, agreement_fraction: float) -> str:
    available = [s for s in sources if s.available]
    if not available:
        return f"No source could evaluate {class_name} for this scene."
    agreeing = ", ".join(s.source for s in available if s.present == fused_present)
    disagreeing = ", ".join(s.source for s in available if s.present != fused_present)
    parts = [
        f"{class_name}: fused verdict is {'PRESENT' if fused_present else 'ABSENT'} "
        f"({len(available)} of {len(sources)} sources available, {agreement_fraction * 100:.0f}% agree)."
    ]
    if agreeing:
        parts.append(f"Agree: {agreeing}.")
    if disagreeing:
        parts.append(f"Disagree: {disagreeing}.")
    unavailable = [s.source for s in sources if not s.available]
    if unavailable:
        parts.append(f"Unavailable: {', '.join(unavailable)}.")
    return " ".join(parts)


def fuse_evidence(
    stack: np.ndarray,
    metadata: dict,
    class_id: ClassId,
    *,
    session: Any = None,
    config: infer.ClassConfig | None = None,
) -> FusionResult:
    """Independently ask four sources whether `class_id` is present in
    `stack`'s scene, and combine them per `SOURCE_WEIGHTS`/`WEIGHTING_NOTE`
    above. `session`/`config` let a caller reuse an already-loaded
    perception.infer session/config (loading either is not cheap); each is
    loaded fresh if omitted.

    Segmentation and classification (from one shared-encoder forward pass,
    sensor='fused') are always available. The spectral index is
    unavailable only when no standard formula applies to `class_id`. The
    SAR branch is unavailable only when `stack` carries no real SAR
    backscatter (see `has_real_sar`) -- it is never guessed at 50/50 in
    that case, it is simply excluded from the fused verdict.
    """
    session = session or infer.load_model()
    config = config or infer.load_class_config()
    class_name = _class_label(class_id)

    result = infer.segment_image(stack, "fused", session=session, config=config)

    sources = [
        _segmentation_evidence(result, class_id, metadata),
        _classification_evidence(result, class_id, config),
        _spectral_evidence(stack, class_id, metadata),
        _sar_evidence(stack, class_id, metadata, session=session, config=config),
    ]

    available = [s for s in sources if s.available]
    if available:
        total_weight = sum(s.weight for s in available)
        fused_confidence = sum(s.weight * s.confidence for s in available) / total_weight
    else:
        fused_confidence = 0.0
    fused_present = fused_confidence >= 0.5

    agreement = {s.source: s.present for s in sources}
    agreement_fraction = (
        sum(1 for s in available if s.present == fused_present) / len(available) if available else 0.0
    )

    return FusionResult(
        class_id=class_id,
        class_name=class_name,
        sources=sources,
        agreement=agreement,
        agreement_fraction=agreement_fraction,
        fused_present=fused_present,
        fused_confidence=fused_confidence,
        summary=_render_summary(class_name, sources, fused_present, agreement_fraction),
    )
