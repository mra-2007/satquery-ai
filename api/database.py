"""SQLite persistence for scene metadata and query/report logs.

Per CLAUDE.md: SQLite, not PostgreSQL; plain files, not Celery. Masks
themselves live as .npy files on disk (api/scenes.py) -- this module only
stores metadata pointing to them, plus a log of past /query results so
GET /report can hand one back later without re-running anything.
"""

import sqlite3
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data" / "api" / "satquery.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scenes (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    gsd_metres REAL NOT NULL,
    sensor TEXT NOT NULL,
    mask_path TEXT NOT NULL,
    classes_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS query_log (
    id TEXT PRIMARY KEY,
    scene_id TEXT NOT NULL,
    query TEXT NOT NULL,
    created_at TEXT NOT NULL,
    response_json TEXT NOT NULL,
    FOREIGN KEY (scene_id) REFERENCES scenes(id)
);
"""


def get_connection() -> sqlite3.Connection:
    """A fresh connection with the schema ensured to exist. SQLite is
    cheap enough to open per call -- this avoids sharing one connection
    across FastAPI's threadpool-executed request handlers."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn
