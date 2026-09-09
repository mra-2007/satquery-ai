"""Session-wide test isolation.

Every test in this suite must run against a throwaway SQLite database and
throwaway mask/stack/upload directories -- never data/api/satquery.db and
data/api/{masks,stacks,uploads}, the real files the live product (and
demo/eval scripts) read and write. Without this, running pytest against a
running (or later-started) instance of the product pollutes its real scene
list with test fixtures -- 64x64 grayscale PNGs, single-pixel change pairs,
severed/misaligned pairs, and so on -- that have no business being in the
real demo/product database. This happened for real: by the time it was
noticed, the production database held 140 scenes, 124 of them test junk
(see git history around the cleanup commit for the one-off script that
repaired it). This fixture exists so it can't happen again.

Session-scoped (not function-scoped) so it patches api.database.DB_PATH and
api.scenes's/api.main's directory constants exactly once, before ANY test
(including a module-scoped fixture like test_api.py's `client`, which
triggers FastAPI's startup lifespan -- ensure_scenes_loaded() and
ensure_demo_change_scene_loaded() -- the moment it's first requested) can
read the real, unpatched paths. A function-scoped fixture would be set up
AFTER a module-scoped one for the same test (broader-scoped fixtures are
always instantiated first), which would be too late here.

data/demo_patches/ and data/oscd/ (the real, read-only source data) are
deliberately NOT redirected -- tests should still exercise the same real
demo patches and OSCD pair the product does; only where derived output
(the database, masks, stacks, uploads, the OSCD PNG extraction) gets
written is redirected.
"""

import pytest

import api.database as database
import api.main as main
import api.scenes as scenes


@pytest.fixture(autouse=True, scope="session")
def _isolated_data_dir(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("satquery_test_data")

    mp = pytest.MonkeyPatch()
    mp.setattr(database, "DB_PATH", tmp_dir / "satquery.db")
    mp.setattr(scenes, "MASKS_DIR", tmp_dir / "masks")
    mp.setattr(scenes, "STACKS_DIR", tmp_dir / "stacks")
    mp.setattr(scenes, "OSCD_EXTRACTED_DIR", tmp_dir / "oscd_extracted")
    mp.setattr(main, "UPLOADS_DIR", tmp_dir / "uploads")
    yield
    mp.undo()
