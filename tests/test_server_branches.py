"""Tests for remaining branches in run_query_safely, paginated, stream, and inner sync bodies."""
import pytest
from unittest.mock import patch, AsyncMock

from tests.conftest import make_ctx
import vertica_mcp.server as server_module


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _small_script():
    return {
        "columns": ["user_id", "name"],
        "probe_rows": [(f"u{i}", f"name{i}") for i in range(3)],
        "page_rows": [(f"u{i}", f"name{i}") for i in range(3)],
        "stream_batches": [[(f"u{i}", f"name{i}") for i in range(3)]],
        "count": 3,
    }

def _large_script(n=1200):
    rows = [(f"u{i}", f"name{i}") for i in range(n)]
    return {
        "columns": ["user_id", "name"],
        "probe_rows": rows,
        "page_rows": rows[:2000],
        "stream_batches": [rows[i:i+500] for i in range(0, n, 500)],
        "count": n,
    }


# ─── run_query_safely: missing client_id ─────────────────────────────────────

@pytest.mark.asyncio
async def test_run_query_safely_no_client_id(make_ctx):
    ctx, manager = make_ctx(_small_script())
    ctx.request_context.client_id = None
    with pytest.raises(RuntimeError, match="Rate limiting requires authenticated"):
        await server_module.run_query_safely(ctx=ctx, query="SELECT 1")


# ─── run_query_safely: rate limit exceeded ───────────────────────────────────

@pytest.mark.asyncio
async def test_run_query_safely_rate_limit_exceeded(make_ctx):
    ctx, manager = make_ctx(_small_script())
    with patch.object(server_module, "_check_rate_limit", return_value=False):
        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            await server_module.run_query_safely(ctx=ctx, query="SELECT 1")


# ─── run_query_safely: invalid / unsafe query ────────────────────────────────

@pytest.mark.asyncio
async def test_run_query_safely_invalid_query(make_ctx):
    ctx, manager = make_ctx(_small_script())
    with pytest.raises(RuntimeError, match="Query validation failed"):
        await server_module.run_query_safely(ctx=ctx, query="DROP TABLE users")


# ─── run_query_safely: auto-LIMIT applied ────────────────────────────────────

@pytest.mark.asyncio
async def test_run_query_safely_auto_limit_applied(make_ctx):
    """SELECT without LIMIT should auto-get LIMIT 1000 wrapper."""
    ctx, manager = make_ctx(_small_script())
    # The wrapped query "SELECT * FROM (...) q LIMIT 1000" counts as SELECT,
    # so probe step runs; result is small so 'done=True' is returned.
    result = await server_module.run_query_safely(
        ctx=ctx, query="SELECT user_id FROM t"
    )
    assert result["ok"] is True
    assert result.get("done") is True


# ─── run_query_safely: small result (< threshold) ────────────────────────────

@pytest.mark.asyncio
async def test_run_query_safely_small_result(make_ctx):
    ctx, manager = make_ctx(_small_script())
    result = await server_module.run_query_safely(
        ctx=ctx, query="SELECT user_id FROM t LIMIT 10"
    )
    assert result["ok"] is True
    assert result["large"] is False
    assert result["done"] is True


# ─── run_query_safely: large result requires confirmation ────────────────────

@pytest.mark.asyncio
async def test_run_query_safely_large_requires_confirmation(make_ctx):
    ctx, manager = make_ctx(_large_script())
    result = await server_module.run_query_safely(
        ctx=ctx,
        query="SELECT user_id FROM t LIMIT 10",
        row_threshold=5,
    )
    assert result["large"] is True
    assert result["requires_confirmation"] is True
    assert "next_step" in result


# ─── run_query_safely: large result with precount ────────────────────────────

@pytest.mark.asyncio
async def test_run_query_safely_large_with_precount(make_ctx):
    script = _large_script()
    script["count"] = 9999
    ctx, manager = make_ctx(script)
    result = await server_module.run_query_safely(
        ctx=ctx,
        query="SELECT user_id FROM t LIMIT 10",
        row_threshold=5,
        precount=True,
    )
    assert result["large"] is True
    assert result["exact_count"] == 9999


# ─── run_query_safely: include_columns=False ─────────────────────────────────

@pytest.mark.asyncio
async def test_run_query_safely_no_columns(make_ctx):
    ctx, manager = make_ctx(_small_script())
    result = await server_module.run_query_safely(
        ctx=ctx,
        query="SELECT user_id FROM t LIMIT 10",
        include_columns=False,
    )
    assert result["ok"] is True
    assert result["columns"] is None


# ─── run_query_safely: proceed=True, mode=page ───────────────────────────────

@pytest.mark.asyncio
async def test_run_query_safely_proceed_page(make_ctx):
    ctx, manager = make_ctx(_small_script())
    result = await server_module.run_query_safely(
        ctx=ctx,
        query="SELECT user_id FROM t LIMIT 10",
        proceed=True,
        mode="page",
    )
    assert "rows" in result or "result" in result


# ─── run_query_safely: proceed=True, mode=stream ─────────────────────────────

@pytest.mark.asyncio
async def test_run_query_safely_proceed_stream(make_ctx):
    ctx, manager = make_ctx(_small_script())
    result = await server_module.run_query_safely(
        ctx=ctx,
        query="SELECT user_id FROM t LIMIT 10",
        proceed=True,
        mode="stream",
    )
    assert "result" in result or "rows" in result


# ─── run_query_safely: proceed=True, unknown mode ────────────────────────────

@pytest.mark.asyncio
async def test_run_query_safely_unknown_mode(make_ctx):
    ctx, manager = make_ctx(_small_script())
    with pytest.raises(RuntimeError, match="Unknown mode"):
        await server_module.run_query_safely(
            ctx=ctx,
            query="SELECT 1 LIMIT 1",
            proceed=True,
            mode="invalid_mode",
        )


# ─── execute_query_paginated ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_query_paginated_basic(make_ctx):
    ctx, manager = make_ctx(_small_script())
    result = await server_module.execute_query_paginated(
        ctx=ctx, query="SELECT user_id FROM t", limit=100, offset=0
    )
    assert "rows" in result
    assert "count" in result
    assert "next_offset" in result
    assert isinstance(result["done"], bool)


@pytest.mark.asyncio
async def test_execute_query_paginated_no_columns(make_ctx):
    ctx, manager = make_ctx(_small_script())
    result = await server_module.execute_query_paginated(
        ctx=ctx,
        query="SELECT user_id FROM t",
        limit=100,
        offset=0,
        include_columns=False,
    )
    assert result["columns"] is None


@pytest.mark.asyncio
async def test_execute_query_paginated_rejects_non_select(make_ctx):
    ctx, manager = make_ctx(_small_script())
    with pytest.raises(RuntimeError, match="Paginated execution only supports SELECT"):
        await server_module.execute_query_paginated(
            ctx=ctx, query="INSERT INTO t VALUES (1)"
        )


@pytest.mark.asyncio
async def test_execute_query_paginated_no_connection(make_ctx):
    ctx, _ = make_ctx(_small_script())
    ctx.request_context.lifespan_context.clear()
    with pytest.raises(RuntimeError, match="No database connection manager available"):
        await server_module.execute_query_paginated(ctx=ctx, query="SELECT 1")


# ─── execute_query_stream ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_query_stream_basic(make_ctx):
    ctx, manager = make_ctx(_small_script())
    fake_stream = {
        "result": [("u1", "name1")],
        "total_rows": 1,
        "truncated": False,
        "size_mb": 0.0,
    }
    with patch("asyncio.to_thread", return_value=fake_stream):
        result = await server_module.execute_query_stream(
            ctx=ctx, query="SELECT user_id FROM t"
        )
    assert "result" in result
    assert "total_rows" in result
    assert "truncated" in result


@pytest.mark.asyncio
async def test_execute_query_stream_no_connection(make_ctx):
    ctx, _ = make_ctx(_small_script())
    ctx.request_context.lifespan_context.clear()
    with pytest.raises(RuntimeError, match="No database connection manager available"):
        await server_module.execute_query_stream(ctx=ctx, query="SELECT 1")


@pytest.mark.asyncio
async def test_execute_query_stream_operation_not_allowed(make_ctx):
    ctx, manager = make_ctx(_small_script())
    with patch.object(manager, "is_operation_allowed", return_value=False):
        with pytest.raises(RuntimeError, match="not allowed"):
            await server_module.execute_query_stream(
                ctx=ctx, query="INSERT INTO t VALUES (1)"
            )


# ─── _validate_connection ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_validate_connection_success(make_ctx):
    ctx, manager = make_ctx(_small_script())
    conn = manager.get_connection()
    result = await server_module._validate_connection(conn)
    assert result is True


@pytest.mark.asyncio
async def test_validate_connection_failure():
    class _BrokenConn:
        def cursor(self):
            raise Exception("broken")
    result = await server_module._validate_connection(_BrokenConn())
    assert result is False


# ─── Metadata tools: no connection branch ────────────────────────────────────

@pytest.mark.asyncio
async def test_get_schema_views_no_connection(make_ctx):
    ctx, _ = make_ctx(_small_script())
    ctx.request_context.lifespan_context.clear()
    with pytest.raises(RuntimeError, match="No database connection manager available"):
        await server_module.get_schema_views(ctx=ctx, schema_name="public")


@pytest.mark.asyncio
async def test_get_table_structure_no_connection(make_ctx):
    ctx, _ = make_ctx(_small_script())
    ctx.request_context.lifespan_context.clear()
    with pytest.raises(RuntimeError, match="No database connection manager available"):
        await server_module.get_table_structure(ctx=ctx, table_name="t")


@pytest.mark.asyncio
async def test_get_table_projections_no_connection(make_ctx):
    ctx, _ = make_ctx(_small_script())
    ctx.request_context.lifespan_context.clear()
    with pytest.raises(RuntimeError, match="No database connection manager available"):
        await server_module.get_table_projections(ctx=ctx, table_name="t")
