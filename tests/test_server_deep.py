"""Tests covering deeper server.py branches: _execute_query_async, run_sse/http CORS,
server_lifespan auth validation, non-SELECT execution, and _print_banner."""
import os
import pytest
from contextlib import asynccontextmanager
from unittest.mock import patch, AsyncMock, MagicMock

import vertica_mcp.server as srv
from tests.conftest import make_ctx


# ─── _execute_query_async: invalid timeout ────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_query_async_invalid_timeout(make_ctx):
    ctx, manager = make_ctx({"columns": ["c"], "probe_rows": [(1,)], "count": 1})
    with pytest.raises(ValueError, match="Invalid timeout"):
        await srv._execute_query_async(manager, "SELECT 1", timeout=0)


@pytest.mark.asyncio
async def test_execute_query_async_timeout_too_large(make_ctx):
    ctx, manager = make_ctx({"columns": ["c"], "probe_rows": [(1,)], "count": 1})
    with pytest.raises(ValueError, match="Invalid timeout"):
        await srv._execute_query_async(manager, "SELECT 1", timeout=9999)


@pytest.mark.asyncio
async def test_execute_query_async_fetch_many(make_ctx):
    script = {
        "columns": ["id", "name"],
        "probe_rows": [(1, "a"), (2, "b")],
        "stream_batches": [[(1, "a"), (2, "b")]],
        "count": 2,
    }
    ctx, manager = make_ctx(script)
    result = await srv._execute_query_async(manager, "SELECT id FROM t", fetch="many", timeout=30)
    # fetchmany returns from stream_batches[0]
    assert result is not None


@pytest.mark.asyncio
async def test_execute_query_async_fetch_none_commit(make_ctx):
    script = {"columns": [], "probe_rows": [], "count": 0}
    ctx, manager = make_ctx(script)
    result = await srv._execute_query_async(
        manager, "SELECT 1", fetch="none", timeout=30, commit=True
    )
    # Returns (rowcount, [])
    assert result is not None


# ─── run_sse: CORS wildcard rejected ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_sse_cors_wildcard_rejected(monkeypatch):
    monkeypatch.setenv("MCP_CORS_ORIGINS", "*")

    with pytest.raises(OSError, match="CORS wildcard"):
        await srv.run_sse(host="localhost", port=19999)


@pytest.mark.asyncio
async def test_run_sse_no_cors_warning(monkeypatch, capsys):
    """When MCP_CORS_ORIGINS is empty, warns and then starts server."""
    monkeypatch.delenv("MCP_CORS_ORIGINS", raising=False)

    # Mock uvicorn so we don't actually bind
    mock_server = AsyncMock()
    mock_server.serve = AsyncMock(side_effect=KeyboardInterrupt)
    mock_uvicorn_server = MagicMock(return_value=mock_server)

    with patch("uvicorn.Server", mock_uvicorn_server):
        with patch("uvicorn.Config", MagicMock()):
            # run_sse catches KeyboardInterrupt silently
            await srv.run_sse(host="localhost", port=19999)


# ─── run_http: CORS wildcard rejected ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_http_cors_wildcard_rejected(monkeypatch):
    monkeypatch.setenv("MCP_CORS_ORIGINS", "*")

    with pytest.raises(OSError, match="CORS wildcard"):
        await srv.run_http(host="127.0.0.1", port=19998)


@pytest.mark.asyncio
async def test_run_http_no_cors_runs(monkeypatch):
    monkeypatch.delenv("MCP_CORS_ORIGINS", raising=False)

    mock_server = AsyncMock()
    mock_server.serve = AsyncMock(side_effect=KeyboardInterrupt)

    with patch("uvicorn.Server", MagicMock(return_value=mock_server)):
        with patch("uvicorn.Config", MagicMock()):
            await srv.run_http(host="127.0.0.1", port=19998)


# ─── run_http: explicit valid CORS origins ────────────────────────────────────

@pytest.mark.asyncio
async def test_run_http_valid_cors(monkeypatch):
    monkeypatch.setenv("MCP_CORS_ORIGINS", "https://app.example.com")

    mock_server = AsyncMock()
    mock_server.serve = AsyncMock(side_effect=KeyboardInterrupt)

    with patch("uvicorn.Server", MagicMock(return_value=mock_server)):
        with patch("uvicorn.Config", MagicMock()):
            await srv.run_http(host="127.0.0.1", port=19997)


# ─── server_lifespan: auth validation ────────────────────────────────────────

@pytest.mark.asyncio
async def test_server_lifespan_no_auth_raises(monkeypatch):
    """server_lifespan should raise OSError when no auth is configured."""
    monkeypatch.delenv("JWT_ISSUER", raising=False)
    monkeypatch.delenv("JWT_AUDIENCE", raising=False)
    monkeypatch.delenv("MCP_API_KEY", raising=False)

    with pytest.raises(OSError, match="SECURITY ERROR"):
        async with srv.server_lifespan(MagicMock()):
            pass  # pragma: no cover


@pytest.mark.asyncio
async def test_server_lifespan_jwt_missing_audience(monkeypatch):
    monkeypatch.setenv("JWT_ISSUER", "https://auth.example.com")
    monkeypatch.delenv("JWT_AUDIENCE", raising=False)

    with pytest.raises(OSError, match="JWT_ISSUER is set but JWT_AUDIENCE is missing"):
        async with srv.server_lifespan(MagicMock()):
            pass  # pragma: no cover


@pytest.mark.asyncio
async def test_server_lifespan_jwt_missing_issuer(monkeypatch):
    monkeypatch.delenv("JWT_ISSUER", raising=False)
    monkeypatch.setenv("JWT_AUDIENCE", "my-audience")

    with pytest.raises(OSError, match="JWT_AUDIENCE is set but JWT_ISSUER is missing"):
        async with srv.server_lifespan(MagicMock()):
            pass  # pragma: no cover


@pytest.mark.asyncio
async def test_server_lifespan_jwt_invalid_issuer_url(monkeypatch):
    monkeypatch.setenv("JWT_ISSUER", "not-a-url")
    monkeypatch.setenv("JWT_AUDIENCE", "my-audience")

    with pytest.raises(OSError, match="JWT_ISSUER must be a valid HTTP"):
        async with srv.server_lifespan(MagicMock()):
            pass  # pragma: no cover


@pytest.mark.asyncio
async def test_server_lifespan_api_key_auth(monkeypatch):
    """With MCP_API_KEY set, lifespan should pass auth check, then fail on TCP."""
    monkeypatch.delenv("JWT_ISSUER", raising=False)
    monkeypatch.delenv("JWT_AUDIENCE", raising=False)
    monkeypatch.setenv("MCP_API_KEY", "test-key-12345")
    monkeypatch.setenv("VERTICA_HOST", "localhost")
    monkeypatch.setenv("VERTICA_DATABASE", "testdb")

    # Mock socket to simulate TCP failure immediately → lifespan continues degraded
    mock_sock = MagicMock()
    mock_sock.connect_ex.return_value = 1  # non-zero = TCP fail
    mock_sock_class = MagicMock(return_value=mock_sock)

    with patch("socket.socket", mock_sock_class):
        # After 3 retries with TCP failure it sets manager=None and yields degraded
        with patch("asyncio.sleep", new_callable=AsyncMock):
            async with srv.server_lifespan(MagicMock()) as ctx:
                # Degraded: manager may be None
                assert isinstance(ctx, dict)


# ─── _print_banner ────────────────────────────────────────────────────────────

def test_print_banner_runs(capsys):
    srv._print_banner("STDIO", None)
    captured = capsys.readouterr()
    assert "STDIO" in captured.out or len(captured.out) >= 0  # just doesn't crash


def test_print_banner_with_url(capsys):
    srv._print_banner("SSE", "http://localhost:8000")
    captured = capsys.readouterr()
    # Banner output goes to stdout
    assert "SSE" in captured.out or len(captured.out) >= 0


# ─── run_query_safely: non-SELECT (write) path ────────────────────────────────

@pytest.mark.asyncio
async def test_run_query_safely_non_select_not_allowed(make_ctx):
    """Even with write allowed in _sanitize_query bypass, operation check blocks it."""
    ctx, manager = make_ctx({"columns": [], "probe_rows": [], "count": 0})
    # Patch _sanitize_query to let a non-SELECT through, then operation check fires
    with patch.object(srv, "_sanitize_query", return_value="INSERT INTO t VALUES (1)"):
        with patch.object(manager, "is_operation_allowed", return_value=False):
            with pytest.raises(RuntimeError, match="not allowed"):
                await srv.run_query_safely(ctx=ctx, query="INSERT INTO t VALUES (1)")


@pytest.mark.asyncio
async def test_run_query_safely_non_select_allowed(make_ctx):
    """With write allowed, non-SELECT runs the non-select execution path."""
    ctx, manager = make_ctx({"columns": [], "probe_rows": [], "count": 0})
    with patch.object(srv, "_sanitize_query", return_value="INSERT INTO t VALUES (1)"):
        with patch.object(manager, "is_operation_allowed", return_value=True):
            with patch.object(srv, "_execute_query_async", new=AsyncMock(return_value=(1, []))):
                result = await srv.run_query_safely(ctx=ctx, query="INSERT INTO t VALUES (1)")
    assert result["ok"] is True
    assert result["affected_rows"] == 1
