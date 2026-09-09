"""Cross-modal analysis: runs perception/infer.py THREE times over the
same raw stack -- sensor_id="optical", "sar", and "fused" -- and reports,
per class, which sensor(s) actually found it. This is the concrete,
computed answer behind a "cross_modal" question (agent/tasks.py's
classify_task label of the same name, e.g. "Does the SAR image confirm
the flooding visible in the optical image?"): three real segmentation
runs, compared, never a single model's guess presented as agreement.

Per CLAUDE.md's "No pixel, no claim": every area number in a
CrossModalFinding comes from evidence/ops.py's size(), run over each of
the three real predicted masks. This module runs inference and
comparison, nothing here is asked of or smoothed by an LLM.

Demo mode: simulate_cloud_cover() zeroes the stack's real optical
channels over the top `cloud_fraction` of the image (60% by default),
leaving the two SAR channels and auxiliary channels untouched -- SAR
penetrates cloud cover in reality, optical does not. Running
cross_modal_analysis(cloud_simulation=True) segments that clouded stack
all three ways: the optical-only path sees the degraded input and should
lose whatever the cloud covers, while the fused path -- given the same
clouded input, but also the intact SAR channels -- should hold up better.
This demonstrates a real, disclosed model behaviour on real data
(data/demo_patches/*.npz, which carry genuine SAR), not a scripted or
guaranteed outcome: cross_modal_analysis() reports whatever the model
actually does with the clouded input, including if fusion doesn't help
as much as expected.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from agent.vocabulary import SEGMENTATION_CLASSES
from evidence import ops
from perception import infer

# The 10 real Sentinel-2 optical band positions in the model's 16-channel
# input (see scripts/demo_real.py's docstring: channel 0 = B02 Blue ...
# channel 9 = B12 SWIR2). Channels 10-11 are the two SAR (VV/VH) bands;
# 12-15 are auxiliary bands. Cloud simulation only ever touches 0-9.
OPTICAL_CHANNEL_INDICES: tuple[int, ...] = tuple(range(10))

DEFAULT_CLOUD_FRACTION = 0.6

_SENSOR_ORDER = ("optical", "sar", "fused")


@dataclass
class CrossModalFinding:
    class_id: int
    class_name: str
    optical_area_ha: float
    sar_area_ha: float
    fused_area_ha: float
    detected_by: tuple[str, ...]  # subset of ("optical", "sar", "fused"), in that order, where area > 0
    attribution: str  # plain-English note on what this detection pattern means


@dataclass
class CrossModalResult:
    findings: list[CrossModalFinding]
    cloud_simulated: bool
    cloud_fraction: float | None
    summary: str
    # Per-sensor USABLE/INSUFFICIENT verdict for the CROSS-MODAL COMPARISON
    # view's three OPTICAL/SAR/FUSED toggles -- see _sensor_status().
    sensor_status: dict[str, str]


_ATTRIBUTION_BY_PATTERN: dict[tuple[bool, bool, bool], str] = {
    # (optical_present, sar_present, fused_present) -> plain-English attribution
    (True, True, True): "detected by all three sensor modes",
    (True, True, False): "optical and SAR both found this alone, but the fused model did not (disagreement)",
    (True, False, True): "optical contribution -- found by optical and the fused model, missed by SAR-only",
    (False, True, True): "SAR contribution -- found by SAR and the fused model, missed by optical-only",
    (True, False, False): "optical-only detection -- not confirmed by SAR or the fused model",
    (False, True, False): "SAR-only detection -- not confirmed by optical or the fused model",
    (False, False, True): "fusion-only detection -- neither single sensor found this alone",
    (False, False, False): "not detected by any sensor mode (should not occur as a finding)",
}


def simulate_cloud_cover(stack: np.ndarray, cloud_fraction: float = DEFAULT_CLOUD_FRACTION) -> np.ndarray:
    """Zero the stack's real optical channels (indices 0-9) over the top
    `cloud_fraction` of the image, leaving SAR (10-11) and auxiliary
    (12-15) channels untouched -- simulating cloud cover an optical
    sensor cannot see through but SAR can. Returns a new array; `stack`
    is never modified in place."""
    if not 0.0 <= cloud_fraction <= 1.0:
        raise ValueError(f"cloud_fraction must be in [0, 1], got {cloud_fraction}")
    clouded = stack.copy()
    _channels, height, _width = stack.shape
    cloud_rows = int(round(height * cloud_fraction))
    for channel in OPTICAL_CHANNEL_INDICES:
        clouded[channel, :cloud_rows, :] = 0.0
    return clouded


def _render_summary(findings: list[CrossModalFinding], cloud_simulated: bool) -> str:
    if not findings:
        return "No land-cover classes were detected by any sensor mode."

    def group(pattern: tuple[str, ...]) -> list[CrossModalFinding]:
        return [f for f in findings if f.detected_by == pattern]

    all_three = group(("optical", "sar", "fused"))
    sar_contribution = group(("sar", "fused"))
    optical_contribution = group(("optical", "fused"))
    fusion_only = group(("fused",))
    optical_only = group(("optical",))
    sar_only = group(("sar",))
    single_mode_conflict = group(("optical", "sar"))

    parts = [f"{len(findings)} class(es) detected across the three sensor modes."]

    def add(label: str, findings_group: list[CrossModalFinding]) -> None:
        if findings_group:
            names = ", ".join(f.class_name for f in findings_group)
            parts.append(f"{label}: {names}.")

    add("Agree across all three", all_three)
    add("SAR contribution (missed by optical-only)", sar_contribution)
    add("Optical contribution (missed by SAR-only)", optical_contribution)
    add("Fusion-only detections", fusion_only)
    add("Optical-only detections (unconfirmed)", optical_only)
    add("SAR-only detections (unconfirmed)", sar_only)
    add("Optical and SAR agree but the fused model disagrees", single_mode_conflict)

    if cloud_simulated:
        parts.append(
            "This run used simulated cloud cover over the optical channels -- "
            "compare the optical and fused area columns above to see whether fusion held up."
        )
    return " ".join(parts)


def _sensor_status(cloud_simulated: bool) -> dict[str, str]:
    """USABLE/INSUFFICIENT verdict per sensor mode, for the CROSS-MODAL
    COMPARISON view's three toggles. This is a statement about the INPUT
    each sensor path actually saw, not a per-finding judgment: when
    cloud_simulation zeroed the real optical channels (simulate_cloud_cover),
    the optical-only path saw degraded input and is marked INSUFFICIENT,
    while SAR -- untouched by the simulated cloud, per its real ability to
    penetrate cloud cover -- and the fused path -- which still had the
    intact SAR channels to fall back on -- are both USABLE. Without cloud
    simulation every sensor saw its real, undegraded input, so all three
    are USABLE."""
    if not cloud_simulated:
        return {"optical": "USABLE", "sar": "USABLE", "fused": "USABLE"}
    return {"optical": "INSUFFICIENT", "sar": "USABLE", "fused": "USABLE"}


def cross_modal_analysis(
    stack: np.ndarray,
    metadata: dict,
    *,
    cloud_simulation: bool = False,
    cloud_fraction: float = DEFAULT_CLOUD_FRACTION,
    session: Any = None,
    config: Any = None,
) -> CrossModalResult:
    """Segment `stack` three ways (optical/sar/fused) and report, per
    class, which sensor(s) found it.

    When `cloud_simulation` is True, simulate_cloud_cover() is applied to
    `stack` first (over `cloud_fraction` of the image) before all three
    runs -- so the optical-only run sees degraded input while the
    SAR-only and fused runs still see real SAR data. `session`/`config`
    let a caller reuse an already-loaded perception.infer session/config
    (loading either is not cheap) -- each is loaded fresh if omitted.
    """
    session = session or infer.load_model()
    config = config or infer.load_class_config()

    working_stack = simulate_cloud_cover(stack, cloud_fraction) if cloud_simulation else stack

    masks = {
        sensor: infer.segment_image(working_stack, sensor, session=session, config=config).mask
        for sensor in _SENSOR_ORDER
    }

    findings: list[CrossModalFinding] = []
    for class_id, class_name in enumerate(SEGMENTATION_CLASSES):
        areas = {sensor: ops.size(masks[sensor], class_id, metadata) for sensor in _SENSOR_ORDER}
        present = {sensor: areas[sensor] > 0 for sensor in _SENSOR_ORDER}
        if not any(present.values()):
            continue

        detected_by = tuple(sensor for sensor in _SENSOR_ORDER if present[sensor])
        findings.append(CrossModalFinding(
            class_id=class_id,
            class_name=class_name,
            optical_area_ha=areas["optical"],
            sar_area_ha=areas["sar"],
            fused_area_ha=areas["fused"],
            detected_by=detected_by,
            attribution=_ATTRIBUTION_BY_PATTERN[(present["optical"], present["sar"], present["fused"])],
        ))

    findings.sort(key=lambda f: f.fused_area_ha, reverse=True)

    return CrossModalResult(
        findings=findings,
        cloud_simulated=cloud_simulation,
        cloud_fraction=cloud_fraction if cloud_simulation else None,
        summary=_render_summary(findings, cloud_simulation),
        sensor_status=_sensor_status(cloud_simulation),
    )
