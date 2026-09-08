"""End-to-end demo on REAL patches with REAL ground truth: loads every
.npz under data/demo_patches/ (each a genuine reBEN Sentinel-1+2 stack plus
its own pixel-level ground-truth class raster), runs perception/infer.py on
all 16 real channels -- no zero-filling, unlike the earlier OSCD-RGB-only
version of this script -- then answers the same four questions through
classify_task -> planner -> validate -> executor against BOTH the
predicted mask and the ground-truth mask, so the two can be compared
side by side. Also reports per-patch pixel accuracy and mIoU of predicted
vs ground truth, and saves both masks as PNGs.

Where these patches come from
------------------------------
data/demo_patches/*.npz are a small local sample of reBEN (Refined
BigEarthNet) patches -- the same source CLAUDE.md / data/download.py says
normally lives only inside the Kaggle notebook, because the full dataset is
too large to store locally. Each file has three arrays:
  - "stack": (17, 120, 120) float16 -- 16 real bands (10 Sentinel-2
    reflectance bands, 2 Sentinel-1 SAR bands, and a few auxiliary bands;
    exactly what models/class_config.json's use_ch/mean/std expect) plus one
    always-zero 17th band we drop before inference. All 16 used channels
    are real pixels, so we run with sensor="fused" and do not zero-fill.
  - "mask": (120, 120) uint8 -- a genuine per-pixel ground-truth class
    raster, not a hand-built one.
  - "labels": the patch's distinct class names, for sanity-checking only.

On class-id order -- no remapping needed
------------------------------------------
An earlier version of this script remapped the ground-truth mask into
models/class_config.json's alphabetically-sorted label order before
comparing it to the prediction, because that order looked plausible and
the two rasters used visibly different integers per class. That remap was
wrong, and it silently compared unrelated classes -- it drove mean mIoU
over these 9 patches from 0.36 down to 0.01. Per the training notebook:
the segmentation target was `y = lut[labels_all[row]]` where `lut` came
from CLASSMAP, and CLASSMAP was the *identity* map {0:0, 1:1, ..., 18:18}
-- no classes were merged for segmentation. So the model's predicted mask
index i corresponds directly to labels_all class id i, which is exactly
the order the ground-truth "mask" arrays already use (SEGMENTATION_LABEL_
ORDER below, reverse-engineered by cross-checking every patch's mask ids
against its bundled "labels" names). class_config.json's `labels` list is
a *different*, alphabetically-sorted vocabulary used only by the model's
separate classification head (scene-level multi-label tags), never by
segmentation -- it must not be used to name mask pixels. Predicted and
ground-truth masks are therefore compared, simplified, and colorized
directly, with no lookup table in between.

Run with the venv's own interpreter:
    venv\\Scripts\\python.exe -m scripts.demo_real
"""

import colorsys
from pathlib import Path

import numpy as np
from PIL import Image

from agent import planner, tasks
from agent.dsl import PlanValidationError, validate
from agent.executor import run as run_plan
from agent.planner import SceneDescriptor, plan_from_query
from agent.tasks import classify_task
from agent.vocabulary import SEGMENTATION_CLASSES as SEGMENTATION_LABEL_ORDER
from evidence.schema import TraceStep
from perception import infer

# Force offline mode -- see scripts/demo_groundtruth.py's docstring for why.
planner.GOOGLE_API_KEY = None
tasks.GOOGLE_API_KEY = None

ROOT_DIR = Path(__file__).resolve().parent.parent
PATCHES_DIR = ROOT_DIR / "data" / "demo_patches"
OUTPUT_DIR = ROOT_DIR / "data" / "demo_patches_output"
GSD_METRES = 10.0  # reBEN's own documented Sentinel-2 resolution

LAND, WATER, FOREST, BUILDING = 0, 1, 2, 3

SCENE = SceneDescriptor(
    layers=["class_raster"],
    classes={LAND: "land", WATER: "water", FOREST: "forest", BUILDING: "building"},
    sensor="sentinel-2",
    gsd_metres=GSD_METRES,
    bbox=(0.0, 0.0, 0.0, 0.0),  # these patches carry no geotransform
)
METADATA = {"gsd_metres": GSD_METRES}

QUESTIONS = [
    "How many water bodies are in this area?",
    "How much forest is there in hectares?",
    "Is there any built-up area near water?",
    "How many buildings are there?",
]

# The segmentation class order -- identity with labels_all (see the module
# docstring's "no remapping needed" note), and NOT the same order as
# class_config.json's `labels` (that one is the classification head's
# alphabetically-sorted vocabulary). Imported from agent/vocabulary.py
# (as SEGMENTATION_CLASSES, aliased here to its old name) so this script
# and the planner's keyword fallback can never drift apart on it.

# SEGMENTATION_LABEL_ORDER index -> simplified scheme. Everything not
# listed here (arable land, pastures, grassland, permanent crops, ...)
# becomes LAND.
RAW_LABEL_TO_SIMPLIFIED = {
    8: FOREST,     # Broad-leaved forest
    9: FOREST,     # Coniferous forest
    10: FOREST,    # Mixed forest
    13: FOREST,    # Transitional woodland, shrub
    15: WATER,     # Inland wetlands
    16: WATER,     # Coastal wetlands
    17: WATER,     # Inland waters
    18: WATER,     # Marine waters
    0: BUILDING,   # Urban fabric
    1: BUILDING,   # Industrial or commercial units
}


def simplify_mask(raw_mask: np.ndarray, nclasses: int) -> np.ndarray:
    lut = np.full(nclasses, LAND, dtype=np.int64)
    for raw_id, simple_id in RAW_LABEL_TO_SIMPLIFIED.items():
        lut[raw_id] = simple_id
    return lut[raw_mask]


def save_colorized_mask(mask: np.ndarray, path: Path, n_classes: int) -> None:
    palette = [
        tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / n_classes, 0.65, 0.95))
        for i in range(n_classes)
    ]
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(palette):
        rgb[mask == class_id] = color
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path)


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Pixel accuracy and mean IoU of `pred` against `gt`, both already in
    SEGMENTATION_LABEL_ORDER's shared class-id space. mIoU averages over
    the classes present in pred union gt only -- classes absent from both
    are excluded rather than counted as a free perfect score."""
    accuracy = float(np.mean(pred == gt))
    classes_present = sorted(set(np.unique(pred).tolist()) | set(np.unique(gt).tolist()))
    ious = []
    for class_id in classes_present:
        pred_c = pred == class_id
        gt_c = gt == class_id
        union = int(np.sum(pred_c | gt_c))
        if union == 0:
            continue
        intersection = int(np.sum(pred_c & gt_c))
        ious.append(intersection / union)
    miou = float(np.mean(ious)) if ious else float("nan")
    return accuracy, miou


def render_answer(trace: list[TraceStep]) -> str:
    parts = []
    limitations = []
    for entry in trace:
        if entry.task == "capability_guardrail":
            limitations.append(entry.output["limitation"])
            continue
        if not entry.task.startswith("execute:"):
            continue
        value = entry.output
        class_name = SCENE.classes.get(entry.parameters.get("class_id", -1))
        if entry.tool == "count":
            parts.append(f"{value} {class_name}(s)")
        elif entry.tool == "size":
            parts.append(f"{value:.2f} ha of {class_name}")
        elif entry.tool in ("presence", "adjacency"):
            parts.append("yes" if value else "no")
        else:
            parts.append(str(value))
    answer = "; ".join(parts)
    for limitation in limitations:
        answer += f"\n      LIMITATION: {limitation}"
    return answer


def compare_answers(query: str, pred_mask: np.ndarray, gt_mask: np.ndarray) -> None:
    print(f"\n{'-' * 78}\nQ: {query}\n{'-' * 78}")

    classification = classify_task(query)
    print(f"  classify_task -> {classification.label!r} (confidence {classification.confidence:.2f})")

    plan = plan_from_query(query, SCENE)
    print(f"  planner       -> {[(s.tool, s.parameters) for s in plan]}")

    try:
        validate(plan)
    except PlanValidationError as exc:
        print(f"  REJECTED before execution: {exc}")
        return
    print("  validate      -> ok")

    for label, mask in (("PREDICTED", pred_mask), ("GROUND TRUTH", gt_mask)):
        _, exec_trace = run_plan(plan, METADATA, mask=mask, classes=SCENE.classes)
        full_trace = list(classification.trace) + list(exec_trace)
        print(f"\n  [{label}] ANSWER: {render_answer(full_trace)}")
        for entry in full_trace:
            print(
                f"      - task={entry.task!r:32s} tool={entry.tool!r:45s} "
                f"params={entry.parameters} output={entry.output} confidence={entry.confidence}"
            )


def process_patch(path: Path, session, config: infer.ClassConfig) -> None:
    print(f"\n{'=' * 78}\nPATCH: {path.name}\n{'=' * 78}")

    data = np.load(path, allow_pickle=True)
    stack, gt_raw = data["stack"], data["mask"]
    real_channels = stack[:16].astype(np.float32)  # drop the always-zero 17th band
    print(f"  stack: {stack.shape} {stack.dtype} -> using {real_channels.shape[0]} real channels "
          f"(no zero-filling); ground-truth labels present: {list(data['labels'])}")

    result = infer.segment_image(real_channels, "fused", session=session, config=config)
    pred_raw = result.mask

    accuracy, miou = compute_metrics(pred_raw, gt_raw)
    print(f"  predicted classes: "
          f"{[SEGMENTATION_LABEL_ORDER[c] for c in sorted(set(pred_raw.flatten().tolist()))]}")
    print(f"  pixel accuracy = {accuracy:.4f}   mIoU = {miou:.4f}")

    stem = path.stem
    pred_png = OUTPUT_DIR / f"{stem}_pred.png"
    gt_png = OUTPUT_DIR / f"{stem}_gt.png"
    save_colorized_mask(pred_raw, pred_png, config.nclasses)
    save_colorized_mask(gt_raw, gt_png, config.nclasses)
    print(f"  saved masks -> {pred_png.relative_to(ROOT_DIR)}, {gt_png.relative_to(ROOT_DIR)}")

    pred_mask = simplify_mask(pred_raw, config.nclasses)
    gt_mask = simplify_mask(gt_raw, config.nclasses)
    for question in QUESTIONS:
        compare_answers(question, pred_mask, gt_mask)


def main() -> None:
    print(__doc__)

    patch_paths = sorted(PATCHES_DIR.glob("*.npz"))
    if not patch_paths:
        raise SystemExit(f"no .npz patches found under {PATCHES_DIR.relative_to(ROOT_DIR)}")
    print(f"Found {len(patch_paths)} patch(es) under {PATCHES_DIR.relative_to(ROOT_DIR)}")

    session = infer.load_model()
    config = infer.load_class_config()
    assert config.nclasses == len(SEGMENTATION_LABEL_ORDER), (
        "class_config.json's nclasses no longer matches SEGMENTATION_LABEL_ORDER"
    )

    for path in patch_paths:
        process_patch(path, session, config)


if __name__ == "__main__":
    main()
