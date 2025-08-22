# tests/conftest.py
import os
import types
import asyncio
import time
import pytest
from typing import Any, List, Dict, Optional

# Fast, deterministic tests
os.environ.setdefault("PYTHONASYNCIODEBUG", "0")
os.environ.setdefault("VERTICA_LAZY_INIT", "1")
os.environ.setdefault("VERTICA_HEALTH_CHECK_INTERVAL", "3600")  # no background probes

# --- Windows: avoid Proactor self-pipe/socketpair weirdness in tests ---
@pytest.fixture(autouse=True, scope="session")
def _windows_selector_event_loop():
    if os.name == "nt":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass
    yield

# Import after env tweaks
import vertica_mcp.connection as vconn
import vertica_mcp.server as server
import uvicorn


# ---------- Global safety patches (autouse) ----------

@pytest.fixture(autouse=True)
def _no_uvicorn_serve(monkeypatch):
    """Prevent any accidental uvicorn server start from blocking tests."""
    async def _noop_serve(self):  # pragma: no cover
        return
    monkeypatch.setattr(uvicorn.Server, "serve", _noop_serve)

@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch):
    """Make any (accidental) sleeps instant to avoid slow tests."""
    async def fast_asyncio_sleep(*_a, **_k):  # pragma: no cover
        return None
    monkeypatch.setattr(asyncio, "sleep", fast_asyncio_sleep)
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)

@pytest.fixture(autouse=True)
def _no_dotenv_walk(monkeypatch):
    """Avoid slow find_dotenv directory walks during imports."""
    monkeypatch.setattr("vertica_mcp.server.find_dotenv", lambda *a, **k: "")


# ---------- Fake Vertica driver ----------

class _FakeCursor:
    """
    Minimal cursor emulation.
    Uses a 'script' dict to decide what to return based on the SQL executed.
    """
    def __init__(self, script: Dict[str, Any]):
        self.script = script or {}
        self._last = None
        self._closed = False
        self._batch_i = 0
        self.description = None  # set for SELECT-like responses

    def execute(self, sql: str):
        sql_upper = sql.strip().upper()

        # Session options: ignore
        if sql_upper.startswith("SET SESSION RUNTIMECAP"):
            self._last = "runtimecap"
            self.description = None
            return

        # Validation probe
        if sql_upper.startswith("SELECT 1"):
            self._last = "probe_1"
            self.description = None
            return

        # Version (unused in our tests)
        if "SELECT VERSION()" in sql_upper:
            self._last = "version"
            self.description = None
            return

        # COUNT(*)
        if "SELECT COUNT(*) FROM" in sql_upper:
            self._last = "count"
            self.description = [("count",)]
            return

        # LIMIT/OFFSET: treat as paginated
        if " LIMIT " in sql_upper:
            if " OFFSET " in sql_upper:
                self._last = "page"
            else:
                self._last = "probe"
            cols = self.script.get("columns", ["c1", "c2"])
            self.description = [(c,) for c in cols]
            return

        # Bare SELECT -> streaming run
        if sql_upper.startswith("SELECT"):
            self._last = "stream"
            cols = self.script.get("columns", ["c1", "c2"])
            self.description = [(c,) for c in cols]
            return

        # Any other (DDL/DML) is "non-select"
        self._last = "nonselect"
        self.description = None

    def fetchall(self) -> List[tuple]:
        if self._last == "probe":
            return list(self.script.get("probe_rows", []))
        if self._last == "page":
            return list(self.script.get("page_rows", self.script.get("probe_rows", [])))
        if self._last in ("stream",):
            return []
        if self._last == "count":
            return []
        return []

    def fetchmany(self, batch_size: int) -> List[tuple]:
        batches: List[List[tuple]] = self.script.get("stream_batches", [])
        if self._batch_i >= len(batches):
            return []
        batch = list(batches[self._batch_i])
        self._batch_i += 1
        return batch

    def fetchone(self) -> Optional[tuple]:
        if self._last == "probe_1":
            return (1,)
        if self._last == "version":
            return (self.script.get("version", "Vertica 24.x"),)
        if self._last == "count":
            if "count" in self.script:
                return (int(self.script["count"]),)
            return (len(self.script.get("probe_rows", [])),)
        return None

    def close(self):
        self._closed = True


class _FakeConnection:
    def __init__(self, script: Dict[str, Any]):
        self.script = script
        self._closed = False
        self._commits = 0
        self._rollbacks = 0

    def cursor(self):
        return _FakeCursor(self.script)

    def commit(self):
        self._commits += 1

    def rollback(self):
        self._rollbacks += 1

    def close(self):
        self._closed = True


# ---------- Helpers / Fixtures ----------

def _make_manager_with_script(monkeypatch, script: Dict[str, Any]) -> vconn.VerticaConnectionManager:
    """
    Create a VerticaConnectionManager with a patched connect() returning a fake connection
    bound to the supplied script.
    """
    # Ensure sane defaults for env
    os.environ.setdefault("VERTICA_HOST", "localhost")
    os.environ.setdefault("VERTICA_PORT", "5433")
    os.environ.setdefault("VERTICA_DATABASE", "VMart")
    os.environ.setdefault("VERTICA_USER", "dbadmin")
    os.environ.setdefault("VERTICA_PASSWORD", "password")
    os.environ.setdefault("VERTICA_LAZY_INIT", "1")  # don't pre-open sockets

    # Patch vertica_python.connect
    def _fake_connect(**kwargs):
        return _FakeConnection(script)

    monkeypatch.setattr(vconn, "vertica_python", types.SimpleNamespace(connect=_fake_connect))

    cfg = vconn.VerticaConfig.from_env()
    mgr = vconn.VerticaConnectionManager()
    mgr.initialize_default(cfg)
    return mgr


class _DummyReqCtx:
    def __init__(self, manager):
        self.lifespan_context = {"vertica_manager": manager}
        self.client_id = "pytest-client"


class _DummyCtx:
    """
    Minimal FastMCP Context-like object with only what your tools use.
    """
    def __init__(self, manager):
        self.request_context = _DummyReqCtx(manager)

    # Logging methods used by the tools (async)
    async def info(self, msg: str):
        return None
    async def warning(self, msg: str):
        return None
    async def error(self, msg: str):
        return None
    async def debug(self, msg: str):
        return None


@pytest.fixture
def make_ctx(monkeypatch):
    """
    Factory fixture to build (ctx, manager) with a given script.
    Usage:
        ctx, manager = make_ctx({"probe_rows": [...], ...})
    """
    def _factory(script: Dict[str, Any]):
        manager = _make_manager_with_script(monkeypatch, script)
        return _DummyCtx(manager), manager
    return _factory


@pytest.fixture
def small_result_script():
    return {
        "version": "Vertica 24.1.0",
        "columns": ["user_id", "name"],
        "probe_rows": [("u1", "John"), ("u2", "Jane"), ("u3", "Bob")],
        "page_rows": [("u1", "John"), ("u2", "Jane")],
        "stream_batches": [[("a", 1), ("b", 2)], [("c", 3)]],
        "count": 3,
    }


@pytest.fixture
def large_result_script():
    rows = [(f"user{i}", f"name{i}") for i in range(0, 1205)]
    return {
        "version": "Vertica 24.1.0",
        "columns": ["user_id", "name"],
        "probe_rows": rows,
        "page_rows": rows[:2000],
        "stream_batches": [rows[i:i+500] for i in range(0, 1500, 500)],
        "count": len(rows),
    }


# Expose modules to tests
@pytest.fixture
def server_module():
    return server


@pytest.fixture
def connection_module():
    return vconn
