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

# Columns added after the initial release, for uploaded (as opposed to
# preloaded demo) scenes: 'kind' distinguishes single/cross_modal/change
# scenes, 'stack_path' holds a saved 16-channel .npy for scenes with no
# original data/demo_patches/*.npz to reconstruct one from, 'before_mask_path'
# is set only for kind='change' (mask_path then holds the AFTER state),
# 'before_stack_path' likewise holds the BEFORE date's saved 16-channel
# input (mirrors 'stack_path', which for a 'change' scene holds the AFTER
# stack) so the CHANGE COMPARISON view can render a true-color preview of
# both dates, not just the after one, 'warnings_json' carries any disclosed
# accuracy caveats (e.g. fewer than 16 real channels), and 'source' is
# 'demo' or 'upload'. Added via ALTER TABLE rather than a schema bump,
# since the existing demo-scene rows and mask files on disk remain
# perfectly valid -- there's nothing to migrate.
_NEW_SCENE_COLUMNS = {
    "kind": "TEXT NOT NULL DEFAULT 'single'",
    "stack_path": "TEXT",
    "before_mask_path": "TEXT",
    "before_stack_path": "TEXT",
    "warnings_json": "TEXT NOT NULL DEFAULT '[]'",
    "source": "TEXT NOT NULL DEFAULT 'demo'",
    # User-supplied acquisition dates for a 'change'-kind pair, as free-text
    # (e.g. "2019-03" or "March 2019") -- NULL whenever the caller didn't
    # provide one (true of every OSCD-seeded demo scene: OSCD's own parquet
    # carries no acquisition-date column). Never fabricated as a default;
    # the CHANGE COMPARISON view shows a plain "BEFORE"/"AFTER" label
    # instead of a made-up date when these are NULL.
    "before_date": "TEXT",
    "after_date": "TEXT",
}


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, declaration in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")


def get_connection() -> sqlite3.Connection:
    """A fresh connection with the schema ensured to exist. SQLite is
    cheap enough to open per call -- this avoids sharing one connection
    across FastAPI's threadpool-executed request handlers."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    _ensure_columns(conn, "scenes", _NEW_SCENE_COLUMNS)
    conn.commit()
    return conn
