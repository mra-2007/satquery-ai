# SatQuery AI

**Agentic vision-language assistant for satellite imagery.**
SIH 2026, problem statement SIH26167 — Team Terrabyte.

> **"No pixel, no claim."**
> The language model never produces an answer. It only classifies the
> question and emits a plan of tool calls. Every number — a count, an
> area in hectares, whether two things are adjacent — is computed by
> deterministic geometry over the model's own predicted mask.

## What this is

SatQuery AI answers natural-language questions about satellite imagery
("How many water bodies are there?", "How much forest is there in
hectares?", "Is there any built-up area near water?") without ever
letting an LLM invent a number. A CNN segments the scene into 19
BigEarthNet land-cover classes; Gemini's only job is to read the question
and decide *which tool to call with which parameters* — the actual
counting, area measurement, and adjacency testing is plain, auditable
Python over the predicted class raster. Every answer ships with a full
execution trace (task → tool → parameters → output → confidence) so it
can be inspected, replayed, and never has to be taken on faith.

On top of the base VQA pipeline, this repo also covers:

- **Change detection** between two dates of the same scene (area
  gained/lost per class, a change mask, focused "how much X changed to
  Y" questions).
- **Cross-modal (optical/SAR) comparison**, including a simulated-cloud
  demo mode showing where radar corroborates optical when the optical
  channel is unusable.
- **Multi-source evidence fusion** — segmentation, the model's own
  classification head, a spectral index (NDVI/NDWI/NDBI), and SAR,
  cross-checked against each other for one class's presence.
- **Disaster-response composite queries** ("how much cropland is
  flooded?") built from the same primitives, not new models.
- **A calibrated confidence score** for every answer, combining
  temperature-scaled perception confidence, test-time-augmentation
  stability, cross-source agreement, resolution suitability, and plan
  validity — with an empirically (not arbitrarily) set refusal threshold.
- A **generative-VLM baseline** (Gemini looking directly at a picture, no
  mask, no geometry) scored against the exact same questions, to make the
  case for deterministic geometry over guessing, honestly, on real
  numbers.

## Architecture

```
Query ─▶ classify_task ─▶ plan_from_query ─▶ validate ─▶ executor.run() ─▶ Evidence
         (agent/tasks.py)  (agent/planner.py)  (agent/dsl.py) (agent/executor.py)
              │                   │                                │
              │                   │                                ▼
         Gemini classifies   Gemini emits a Plan            evidence/ops.py,
         the task type       (JSON tool calls) —            evidence/change.py,
         (vqa/caption/       validated against               evidence/fusion.py,
         change/cross_modal/  agent/registry.py's            confidence/engine.py
         metadata/            tool schemas before             — deterministic
         conversational)      anything runs                   geometry + a
                                                                calibrated
                                                                confidence score
```

| Layer | Directory | Role |
|---|---|---|
| Perception | `perception/` | ONNX Runtime (CPU-only) inference: a 19-class segmentation head + a scene-level classification head, sharing one encoder. Tiles and feather-blends anything larger than the model's native 120×120 input. |
| Reasoning | `agent/` | `tasks.py` (task classification), `planner.py` (query → typed Plan), `dsl.py` (Plan validation/execution), `registry.py` (tool schemas + the GSD-derived capability table), `guardrail.py` (degrades an uncountable-at-this-resolution `count` to a `size`), `vocabulary.py` (natural-language noun → real class(es), many-to-many where needed), `executor.py` (wires a Plan to real tool implementations and the confidence engine). |
| Geometry | `evidence/` | `ops.py` (count/size/presence/adjacency/intersect/buffer), `change.py` (bi-temporal change detection), `fusion.py` (multi-source evidence fusion), `metadata.py` (scene-record facts), `schema.py` (the `Evidence` JSON contract every answer returns). |
| Confidence | `confidence/` | `engine.py` (the five-signal calibrated confidence score), `calibration.py` (temperature scaling, fit against real ground-truth patches). |
| Raster I/O | `raster_io/` | Validates and reads uploaded imagery; GSD is always read from metadata, never hardcoded. |
| Tools | `tools/` | `caption.py`, `grounding.py`, `cross_modal.py`, `verifier.py` (independently re-checks every caption's claims against the real mask), `conversational.py` (greetings/off-topic). |
| API | `api/` | FastAPI app: `/query`, `/upload`, `/scenes`, `/report`, plus per-scene image/mask/legend/change/cross-modal endpoints. SQLite for scene/query metadata, plain files for masks/stacks — no Postgres, no Celery. |
| Frontend | `web/` | React + TypeScript + OpenLayers. Landing page, ANALYZE / CHANGE COMPARISON / CROSS-MODAL COMPARISON views, execution-trace drawer, confidence bars — all rendering the `Evidence` contract generically. |
| Eval | `eval/` | `run_benchmark.py` (BigEarthNet.txt), `run_rsvqa.py` (RSVQA-LR), `run_vlm_baseline.py` (the generative-VLM comparison) — see results below. |
| Scripts | `scripts/` | `demo_real.py`/`demo_groundtruth.py` (offline demo runs), `fit_calibration.py` (fits confidence temperature scaling), `precompute.py` (warms the plan/task cache for the ANALYZE view's example queries). |

### Hard constraints (see `CLAUDE.md`)

CPU-only inference (ONNX Runtime, provider forced, never left to
auto-select a GPU); Python 3.11 in `./venv`; Windows/cmd; SQLite + plain
files, no Postgres/Celery; OpenLayers on the frontend; GSD always read
from image metadata; every tool call recorded in a typed execution trace
with a confidence.

## Benchmark results

Full detail, methodology, and every disclosed caveat: **`eval/RESULTS.md`**.
Reproduce with `venv\Scripts\python.exe -m eval.run_benchmark` /
`-m eval.run_rsvqa`.

### BigEarthNet.txt (654 in-scope questions, real 16-channel Sentinel-1+2 patches)

Scored twice — against the model's real **predicted** mask (what this
system actually answers with) and against the **ground-truth** mask (the
reasoning layer's own ceiling, isolating segmentation error from
reasoning error):

| Category | N | Predicted-mask accuracy | Ground-truth-mask accuracy |
|---|---|---|---|
| presence | 117 | 72.6% | 100.0% |
| count | 109 | 44.0% | 84.4% |
| area | 113 | 76.1% | 100.0% |
| adjacency | 315 | 60.6% | 99.4% |
| **Overall** | 654 | **62.7%** | **97.1%** |

The reasoning layer (question parsing → `classify_task` → planner →
`validate` → executor) is essentially correct on its own (97.1% against
perfect segmentation); the 34-point drop to 62.7% is real segmentation
error from the model itself, largest on `count` (component-count
questions are the most sensitive to a handful of misclassified pixels
fragmenting or merging a region).

### RSVQA-LR (200 questions, degraded input — 13 of 16 channels zero-filled, no real reflectance)

| Question type | N | Accuracy |
|---|---|---|
| presence | 48 | 60.4% |
| count | 74 | 10.8% |
| comparison | 75 | 52.0% |
| rural_urban | 3 | 100.0% |
| **Overall** | 200 | **39.5%** |

Read as a stress test on badly-degraded input (an already-rendered RGB
PNG, not a real Sentinel-2 stack), not a representative measurement —
`eval/RESULTS.md` has the full caveat.

## VLM baseline: deterministic geometry vs. a generative model guessing

`eval/run_vlm_baseline.py` sends Gemini a true-color rendering of the
same scene and the same raw question text (no mask, no geometry — a
direct "look at the picture and answer" baseline, exactly what this
project exists to beat), scored on the *identical* 100-question sample
this system's own predicted-mask pipeline is scored on:

| Category | N | SatQuery AI (predicted mask) | Gemini VLM baseline |
|---|---|---|---|
| presence | 17 | 88.2% | 52.9% |
| count | 14 | 42.9% | 57.1% |
| area | 20 | 80.0% | 75.0% |
| adjacency | 49 | 46.9% | 40.8% |
| **Overall** | 100 | **60.0%** | **52.0%** |

SatQuery AI wins clearly on presence and area — deterministic geometry
reliably resolves what's present and how much of it there is. The VLM
edges ahead on `count`, unsurprisingly: `evidence/ops.py`'s connected-
component count is a fundamentally different quantity from a human's
(or an LLM's) object count, the same gap `eval/run_rsvqa.py`'s own
bin-based count adapter exists to work around. Full per-question detail:
`eval/vlm_baseline_results.csv`.

## Setup

**Requirements:** Python 3.11, Node.js (for the frontend), a Google
Gemini API key.

```cmd
:: Backend
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt

:: Frontend
cd web
npm install
cd ..
```

Create a `.env` file in the repo root (never committed — see
`.gitignore`):

```
GOOGLE_API_KEY=your-gemini-api-key-here
```

Without a key configured, every Gemini-backed step (task classification,
planning, caption smoothing) falls back to a deterministic keyword
parser instead of failing — the system still works fully offline, just
with a coarser understanding of free-form phrasing. See `agent/tasks.py`
and `agent/planner.py`'s own module docstrings for the exact fallback
behaviour.

## Running it

**Backend** (from the repo root, venv not activated — always invoke the
venv's own interpreter directly, per `CLAUDE.md`):

```cmd
venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

Then open `http://127.0.0.1:8000/docs` for the interactive API, or point
the frontend at it.

**Frontend** (from `web/`):

```cmd
npm run dev
```

Opens the landing page at the printed local URL; demo scenes (real
BigEarthNet patches) are loaded automatically on backend startup and are
immediately queryable from the ANALYZE view's example chips.

**Tests:**

```cmd
venv\Scripts\python.exe -m pytest -q
```

**(Optional) warm the plan/task cache** for the ANALYZE view's example
queries, so the first live demo doesn't wait on a Gemini round trip:

```cmd
venv\Scripts\python.exe -m scripts.precompute
```

**(Optional) refit confidence calibration** against the real ground-truth
patches in `data/bench_patches/` (only needed after a model change):

```cmd
venv\Scripts\python.exe -m scripts.fit_calibration
```
