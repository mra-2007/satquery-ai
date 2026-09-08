# SatQuery AI

Agentic vision-language assistant for satellite imagery.
SIH 2026, problem statement SIH26167. Team Terrabyte.

## Core principle
The language model NEVER produces the answer. It only
(1) classifies the task and (2) emits a plan of tool calls.
All numbers -- counts, areas, adjacency -- are COMPUTED by
deterministic geometry in evidence/ops.py. Never by the LLM.
Deck slogan: "No pixel, no claim."

## Hard constraints
- No GPU. CPU inference only, via ONNX Runtime.
- Python 3.11 in ./venv. The venv is NOT activated --
  always run scripts with venv\Scripts\python.exe directly.
- Windows. Use cmd syntax, never PowerShell.
- The io package is named raster_io -- `io` collides with a
  frozen CPython stdlib module and cannot be a package.
- Minimal dependencies.
- No PostgreSQL, no Celery. SQLite and plain files.
- Frontend uses OpenLayers (not Mapbox, not MapLibre).
- Every area calculation reads GSD from image metadata.
  NEVER hardcode 10 m.
- Every tool call is recorded in an execution trace:
  task, tool, parameters, output, confidence.
- Must work offline: cache LLM responses, no external tiles.
- LLM is Google Gemini via GOOGLE_API_KEY in .env.

## Key numbers
- Sentinel patch: 120x120 px at 10 m = 1.2 x 1.2 km
- 144 hectares per patch, 100 square metres per pixel
- Minimum resolvable object size = 2.5 x GSD

## Style
Tests alongside code. Small functions. Type hints.