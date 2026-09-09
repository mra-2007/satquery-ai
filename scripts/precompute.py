"""Pre-warms agent.tasks' task-classification cache and agent.planner's
plan cache with SatQuery AI's five representative ANALYZE-view queries
against the default demo scene, using REAL, live Gemini calls -- then
verifies, with the network genuinely severed at the socket layer (not
merely GOOGLE_API_KEY unset), that all five still resolve correctly end
to end through the real FastAPI app (api/main.py's own /query endpoint,
called exactly the way a real client would call it).

Why both caches, not just "the plan cache"
------------------------------------------------
api/main.py's query_scene() calls BOTH classify_task() (agent/tasks.py)
and plan_from_query() (agent/planner.py) for every live query -- two
separate Gemini calls, two separate disk caches (agent/tasks.py's own
module docstring: "Unlike the planner's cache, this one is keyed by the
question alone -- task classification doesn't depend on which scene the
question is about."). Warming only the plan cache would still leave
classify_task() hitting the network (or falling back) on every request;
both are warmed here so the real end-to-end /query call needs neither.

The "default scene" is scene-content-based, not scene-id-based
--------------------------------------------------------------------
Every single-kind demo scene api.scenes.ensure_scenes_loaded() creates
shares the exact same classes (the full 19-class agent.vocabulary.
SEGMENTATION_CLASSES), sensor ("fused"), gsd_metres (10.0), and bbox
placeholder ((0,0,0,0)) -- see that function's own _insert_scene() calls.
agent.planner's cache key is a hash of the query plus the SceneDescriptor's
own JSON, not the scene_id, so warming it once with that shared descriptor
already covers every demo single scene, not just one. The verification
step below still resolves and hits one concrete scene_id through the real
API (the first demo 'single' scene GET /scenes returns), so "end to end
through the API" is genuinely exercised, not simulated.

Targeted cache eviction before warming
-------------------------------------------
tests/test_api.py's own `offline_mode` fixture deliberately forces
GOOGLE_API_KEY=None and then runs several of these SAME five queries
against a real demo scene -- through the exact same production code path,
writing to these exact same default cache files. Running the test suite
therefore already leaves keyword-fallback-sourced entries (confidence
0.4/0.75, the fallback's own fixed values) cached under these five
queries' exact cache keys. Both caches check the disk FIRST, unconditionally,
before ever considering the network -- so without eviction this script
would just silently read back that stale, low-quality fallback data and
never call Gemini at all, defeating the entire point of "warm the cache
with real output". The five (and only the five) relevant cache files are
therefore deleted before warming, not the caches wholesale.

Verifying with the network truly disabled
----------------------------------------------
GOOGLE_API_KEY=None only proves "no key was configured" -- both
classify_task() and plan_from_query() already have their own try/except
fallback to a deterministic keyword parser on ANY exception, key-related
or not, so that alone can't distinguish "the cache satisfied this
request" from "the network was unreachable and it quietly, correctly,
fell back anyway" (also correct behavior, but not what this script is
proving). network_disabled() below monkeypatches socket.socket.connect
itself to raise -- library-agnostic, catching Gemini's own httpx-based
client and anything else -- so a genuine cache hit is verified by its
actual mechanism (no network attempted at all) rather than inferred from
the absence of a configured key. FastAPI's TestClient talks to the ASGI
app in-process (no real socket), so it works normally under this block.

Run with the venv's own interpreter:
    venv\\Scripts\\python.exe -m scripts.precompute
"""

import socket
from contextlib import contextmanager
from pathlib import Path

from agent import planner, tasks
from agent.dsl import validate
from agent.planner import SceneDescriptor, plan_from_query
from agent.tasks import classify_task
from agent.vocabulary import SEGMENTATION_CLASSES

ROOT_DIR = Path(__file__).resolve().parent.parent
GSD_METRES = 10.0  # every demo patch's own documented resolution
SENSOR = "fused"   # api.scenes.SENSOR -- every demo single scene's own sensor id

# Mirrors web/src/components/CommandBar.tsx's ANALYZE_EXAMPLES[:5] verbatim
# -- the ANALYZE view's own first five example chips. Deliberately
# representative of five different code paths: count (water bodies),
# size/area (forest hectares), adjacency (built-up near water), a
# capability-guardrail-degraded count (buildings -- always too fine at
# 10 m GSD, see agent/guardrail.py), and caption (describe this scene).
# There is no automatic cross-language sync with CommandBar.tsx -- if that
# list changes, update this one too, the same way scripts/demo_real.py's
# own QUESTIONS list already duplicates (rather than imports) its first
# four entries verbatim.
REPRESENTATIVE_QUERIES = [
    "How many water bodies are there?",
    "How much forest is there in hectares?",
    "Is there any built-up area near water?",
    "How many buildings are there?",
    "Describe this scene",
]


def default_scene_descriptor() -> SceneDescriptor:
    """The exact SceneDescriptor api/main.py's query_scene() builds for
    ANY single-kind demo scene (see this module's own docstring) -- no
    scene_id needed at all to warm the cache every one of them shares."""
    return SceneDescriptor(
        layers=["class_raster"],
        classes=dict(enumerate(SEGMENTATION_CLASSES)),
        sensor=SENSOR,
        gsd_metres=GSD_METRES,
        bbox=(0.0, 0.0, 0.0, 0.0),
    )


def _evict_stale_entries(scene: SceneDescriptor) -> None:
    """Deletes just these five queries' own task-cache and plan-cache
    files, if present -- see this module's docstring on why a stale
    keyword-fallback entry from the test suite would otherwise mask a
    genuine re-warm. Never touches any other cache entry."""
    evicted = 0
    for query in REPRESENTATIVE_QUERIES:
        task_path = tasks.DEFAULT_CACHE_DIR / f"{tasks._cache_key(query)}.json"
        plan_path = planner.DEFAULT_CACHE_DIR / f"{planner._cache_key(query, scene)}.json"
        for path in (task_path, plan_path):
            if path.exists():
                path.unlink()
                evicted += 1
    print(f"Evicted {evicted} stale cache file(s) for the five representative queries "
          f"(if any existed from a prior offline test run) ...\n")


def warm_cache(scene: SceneDescriptor) -> dict[str, dict]:
    """Calls classify_task() and plan_from_query() for real, with a live
    Gemini client -- writing both disk caches as an ordinary side effect
    of a normal call, exactly what a real live /query request already
    does. Requires GOOGLE_API_KEY: there is nothing genuine to warm the
    cache WITH otherwise. Returns {query: {label, plan}} for
    verify_offline() below to compare the offline re-run against."""
    if tasks.GOOGLE_API_KEY is None or planner.GOOGLE_API_KEY is None:
        raise SystemExit(
            "GOOGLE_API_KEY is not set -- warming the cache with REAL Gemini "
            "output requires a live call; there is nothing to warm it with otherwise."
        )

    print(f"Warming the cache for {len(REPRESENTATIVE_QUERIES)} representative "
          f"quer{'y' if len(REPRESENTATIVE_QUERIES) == 1 else 'ies'} (live Gemini calls) ...\n")
    warmed: dict[str, dict] = {}
    for query in REPRESENTATIVE_QUERIES:
        classification = classify_task(query)
        plan = plan_from_query(query, scene)
        validate(plan)

        task_source = classification.trace[0].parameters["source"]
        print(f"  {query!r}")
        print(f"    classify_task    -> {classification.label!r} "
              f"(confidence {classification.confidence:.2f}, source={task_source!r})")
        print(f"    plan_from_query  -> {[(s.tool, s.parameters) for s in plan]}")
        if task_source != "gemini":
            raise SystemExit(
                f"classify_task did not reach Gemini for {query!r} (source={task_source!r}) -- "
                "cannot warm the cache with real output; check GOOGLE_API_KEY / network."
            )
        warmed[query] = {"label": classification.label, "plan": plan.model_dump()}
    print()
    return warmed


# --- offline verification: network genuinely severed, not just no key ------


class NetworkDisabledError(RuntimeError):
    """Raised in place of any real, non-loopback network connection during
    offline verification -- so a silent fall-through to the keyword parser
    (which would ALSO return a working answer, and could be mistaken for
    proof the cache worked) can never be confused with an actual cache
    hit. If this ever fires below, the cache did NOT satisfy the request."""


def _is_loopback(address: object) -> bool:
    """True for 127.0.0.1/::1/localhost, or anything that isn't a plain
    (host, port, ...) network address at all (e.g. an AF_UNIX path) --
    loopback and non-IP connections aren't "the network" in the sense
    being blocked here, and asyncio's own Windows ProactorEventLoop opens
    a real loopback socket pair for internal wakeup plumbing on every
    event loop it starts (irrespective of this script), which must keep
    working under network_disabled() or FastAPI's TestClient itself can't
    even start."""
    if isinstance(address, tuple) and address:
        host = str(address[0])
        return host in ("127.0.0.1", "::1", "localhost") or host.startswith("127.")
    return True


@contextmanager
def network_disabled():
    """Blocks only REMOTE connections (anything not loopback) -- a real,
    meaningful block on reaching Gemini's servers, while leaving local
    machinery (asyncio's own internals, FastAPI's in-process TestClient)
    untouched. See this module's docstring for why GOOGLE_API_KEY=None
    alone wouldn't prove the same thing."""
    original_connect = socket.socket.connect

    def _guarded_connect(self, address, *args, **kwargs):
        if not _is_loopback(address):
            raise NetworkDisabledError(
                f"a real network connection to {address!r} was attempted during offline verification"
            )
        return original_connect(self, address, *args, **kwargs)

    socket.socket.connect = _guarded_connect
    try:
        yield
    finally:
        socket.socket.connect = original_connect


def verify_offline(scene: SceneDescriptor, warmed: dict[str, dict]) -> None:
    """With the network hard-blocked: (1) re-calls plan_from_query()
    directly and asserts it reproduces the exact plan warm_cache() cached
    -- the direct proof that the plan cache itself, not guardrail
    rewriting or anything downstream, is what's being verified; then (2)
    hits the real API's /query endpoint (api/main.py, via FastAPI's
    TestClient -- in-process ASGI, no real socket, so it isn't itself
    blocked by network_disabled()) for all five queries against the first
    real demo scene, asserting HTTP 200 and that classify_task's own trace
    entry shows source == "cache"."""
    # Imported here, not at module level, so nothing about warm_cache()
    # above (which needs the network) can ever run after network_disabled()
    # takes effect below.
    from fastapi.testclient import TestClient

    from api.main import app

    print("Verifying the plan cache directly, with the network hard-blocked ...\n")
    with network_disabled():
        for query in REPRESENTATIVE_QUERIES:
            plan = plan_from_query(query, scene)
            cached_plan = warmed[query]["plan"]
            assert plan.model_dump() == cached_plan, (
                f"{query!r}: offline plan {plan.model_dump()} != the plan warm_cache() "
                f"cached {cached_plan} -- the plan cache did not satisfy this request."
            )
            print(f"  OK  {query!r} -> {[(s.tool, s.parameters) for s in plan]}")
    print("\nAll five plans reproduced exactly from cache, zero network access attempted.\n")

    print("Verifying all five queries end to end through the real API (/query), "
          "network still hard-blocked ...\n")
    with network_disabled():
        with TestClient(app) as client:  # startup lifespan: local ONNX inference + SQLite only, no network
            demo_scene_id = next(
                s["id"] for s in client.get("/scenes").json()
                if s["kind"] == "single" and s["source"] == "demo"
            )
            print(f"  default scene: {demo_scene_id}\n")

            for query in REPRESENTATIVE_QUERIES:
                response = client.post("/query", json={"scene_id": demo_scene_id, "query": query})
                assert response.status_code == 200, f"{query!r} -> HTTP {response.status_code}: {response.text}"
                body = response.json()
                trace = body["evidence"]["execution_trace"]

                classify_step = next(s for s in trace if s["task"] == "classify_task")
                task_source = classify_step["parameters"]["source"]
                assert task_source == "cache", (
                    f"{query!r}: classify_task did not hit its cache (source={task_source!r}) through "
                    "the real API, with the network blocked -- it silently fell back to the keyword "
                    "classifier instead of proving the cache."
                )

                value, units = body["evidence"]["value"], body["evidence"]["units"]
                print(f"  OK  {query!r}")
                print(f"      -> {value!r}{f' {units}' if units else ''}  "
                      f"(classify_task source={task_source!r})")
    print("\nAll five representative queries answered correctly end to end through "
          "the real API, with zero network access.")


def main() -> None:
    print(__doc__)
    scene = default_scene_descriptor()
    _evict_stale_entries(scene)
    warmed = warm_cache(scene)
    verify_offline(scene, warmed)


if __name__ == "__main__":
    main()
