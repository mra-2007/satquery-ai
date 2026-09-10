# syntax=docker/dockerfile:1

# --- Stage 1: build the React frontend (web/) ---------------------------
FROM node:20-slim AS frontend-build

WORKDIR /app/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build
# -> /app/web/dist: the exact static bundle `npm run build` produces
# locally, api/main.py serves this itself once it's present (see its
# own "Serve the built React frontend" section) -- no separate web
# server, one port for both frontend and backend.


# --- Stage 2: the FastAPI backend + the built frontend -------------------
FROM python:3.11-slim AS backend

# System dependencies for GDAL/rasterio/geopandas (raster_io/, evidence/,
# and perception/ all read real georeferenced imagery through rasterio).
# gdal-bin/libgdal-dev/libgeos-dev/libproj-dev cover the C libraries these
# wheels link against; build-essential covers a source build for anything
# that doesn't ship a prebuilt wheel for this platform.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin \
        libgdal-dev \
        libgeos-dev \
        libproj-dev \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies first, in their own layer, so an app-code-only
# change doesn't force a full dependency reinstall on rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code -- explicit, one directory at a time (not `COPY . .`),
# so it's obvious exactly what ships and nothing unexpected rides along.
COPY agent/ agent/
COPY api/ api/
COPY confidence/ confidence/
COPY evidence/ evidence/
COPY perception/ perception/
COPY raster_io/ raster_io/
COPY tools/ tools/

# The real trained model + its class config (per CLAUDE.md: never remapped,
# never regenerated -- this exact ONNX file and this exact config, shipped
# as-is) and the real BigEarthNet demo patches api.scenes.ensure_scenes_loaded()
# seeds its queryable demo scenes from at startup.
COPY models/ models/
COPY data/demo_patches/ data/demo_patches/

# The frontend build from stage 1 -- api/main.py serves it directly when
# web/dist exists (see api/main.py's own "Serve the built React frontend"
# section), so this single image serves both frontend and backend.
COPY --from=frontend-build /app/web/dist web/dist

# GOOGLE_API_KEY is read from the environment at runtime (agent/planner.py,
# agent/tasks.py, tools/caption.py) -- deliberately NOT set here, so it is
# never baked into the image. Hugging Face Spaces injects it as a runtime
# secret; `docker run -e GOOGLE_API_KEY=...` does the same locally. Missing
# entirely, the app still runs -- every Gemini-backed step falls back to a
# deterministic keyword parser instead of failing (see agent/planner.py's
# own module docstring).

# 7860 is Hugging Face Spaces' own default port for a Docker Space.
EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
