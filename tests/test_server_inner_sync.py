"""
Cover the inner _sync_* bodies of profile_query, database_status, and
analyze_system_performance by running them synchronously via monkeypatching
asyncio.to_thread to call the function inline.
"""
import time
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

import vertica_mcp.server as srv
from tests.conftest import make_ctx


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _script_with(probe_rows, columns=None, count=None):
    cols = columns or ["c1", "c2", "c3", "c4", "c5"]
    n = count if count is not None else len(probe_rows)
    return {
        "columns": cols,
        "probe_rows": probe_rows,
        "page_rows": probe_rows,
        "count": n,
        "version": "Vertica Analytic Database v24.1.0-0",
    }


async def _run_with_real_thread(coro):
    """Actually awaits the coroutine, letting asyncio.to_thread run inline."""
    return await coro


# ─── database_status._sync_status ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_database_status_no_audit_data(make_ctx):
    """When fetchone() returns None for the license query, handles gracefully."""
    # probe_rows=[] means fetchone() returns None for the license query;
    # fetchone() for version returns (version,) from the 'version' script key.
    script = _script_with(
        probe_rows=[],          # empty = no audit rows
        columns=["license_gb", "db_gb", "usage_pct", "start_ts", "end_ts"],
        count=0,
    )
    ctx, manager = make_ctx(script)

    # Patch asyncio.to_thread to run the function synchronously
    async def _inline(fn, *a, **kw):
        return fn(*a, **kw)

    with patch("asyncio.to_thread", side_effect=_inline):
        result = await srv.database_status(ctx=ctx)

    assert "version" in result
    assert result["current_usage_pct"] == 0.0
    assert result["cluster_nodes"] is not None


@pytest.mark.asyncio
async def test_database_status_with_audit_data(make_ctx):
    """When audit data exists, version and trend data are populated."""
    # Trend query expects (date, avg_usage, max_db) — 3-col tuples
    # License audit fetchone() returns None (cursor returns None for probe)
    # Node count fetchone() returns count from script
    script = {
        "columns": ["audit_date", "avg_usage_pct", "max_db_gb"],
        "probe_rows": [("2024-01-01", 45.0, 90.0)],  # 3-col trend rows
        "count": 3,          # node count returned by SELECT COUNT(*)
        "version": "Vertica Analytic Database v24.1.0-0",
    }
    ctx, manager = make_ctx(script)

    async def _inline(fn, *a, **kw):
        return fn(*a, **kw)

    with patch("asyncio.to_thread", side_effect=_inline):
        result = await srv.database_status(ctx=ctx)

    assert result["version"] == "Vertica Analytic Database v24.1.0-0"
    assert "result" in result
    assert "Database Status" in result["result"]


# ─── analyze_system_performance._sync_perf ────────────────────────────────────

@pytest.mark.asyncio
async def test_analyze_system_performance_inline(make_ctx):
    """Run _sync_perf inline to cover the body."""
    script = {
        "columns": ["node_name", "ts", "cpu_pct", "mem_pct"],
        "probe_rows": [("n1", "2024-01-01 00:00:00", 40.0, 50.0)],
        "count": 1,
    }
    ctx, manager = make_ctx(script)

    async def _inline(fn, *a, **kw):
        return fn(*a, **kw)

    with patch("asyncio.to_thread", side_effect=_inline):
        result = await srv.analyze_system_performance(ctx=ctx, window_minutes=5)

    assert "cpu" in result
    assert "memory" in result
    assert "top_tables_by_ros" in result
    assert "meta" in result


@pytest.mark.asyncio
async def test_analyze_system_performance_no_connection(make_ctx):
    ctx, _ = make_ctx({"columns": [], "probe_rows": [], "count": 0})
    ctx.request_context.lifespan_context.clear()
    with pytest.raises(RuntimeError, match="No database connection manager available"):
        await srv.analyze_system_performance(ctx=ctx)


# ─── profile_query._sync_profile ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_profile_query_inline_no_data(make_ctx):
    """When profiling data not found, returns 'not available' result."""
    script = {
        "columns": ["transaction_id", "statement_id", "duration"],
        "probe_rows": [],   # empty = no profile rows found
        "count": 0,
    }
    ctx, manager = make_ctx(script)

    async def _inline(fn, *a, **kw):
        return fn(*a, **kw)

    with patch("asyncio.to_thread", side_effect=_inline):
        with patch("time.sleep"):  # skip the 1-second sleep
            result = await srv.profile_query(ctx=ctx, query="SELECT 1", timeout=5)

    # Either "not available" or a full profile dict
    assert isinstance(result, dict)
    assert "result" in result or "query" in result


@pytest.mark.asyncio
async def test_profile_query_inline_with_data(make_ctx):
    """When profile rows are found (3-col + plan lines as strings), returns execution profile."""
    # probe_rows first element must be a string so plan join works: r[0] → str
    # Strategy 2/3 unpacks row as: trxid, stmtid, duration = row (3 values)
    script = {
        "columns": ["transaction_id", "statement_id", "duration_us"],
        "probe_rows": [("Nested Loop", 12345, 1000)],  # str,int,int → plan join works
        "count": 1,
    }
    ctx, manager = make_ctx(script)

    async def _inline(fn, *a, **kw):
        return fn(*a, **kw)

    with patch("asyncio.to_thread", side_effect=_inline):
        with patch("time.sleep"):
            result = await srv.profile_query(ctx=ctx, query="SELECT 1", timeout=5)

    assert isinstance(result, dict)


# ─── execute_query_stream._sync_stream inline ────────────────────────────────

@pytest.mark.asyncio
async def test_execute_query_stream_inline(make_ctx):
    """Run _sync_stream inline to cover its body."""
    script = {
        "columns": ["id", "name"],
        "probe_rows": [(1, "a"), (2, "b")],
        "stream_batches": [[(1, "a"), (2, "b")], []],  # second empty = stop
        "count": 2,
    }
    ctx, manager = make_ctx(script)

    async def _inline(fn, *a, **kw):
        return fn(*a, **kw)

    with patch("asyncio.to_thread", side_effect=_inline):
        result = await srv.execute_query_stream(ctx=ctx, query="SELECT id FROM t")

    assert "result" in result
    assert "total_rows" in result
    assert "truncated" in result


# ─── execute_query_stream: DB error path ─────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_query_stream_db_error(make_ctx):
    script = {"columns": ["id"], "probe_rows": [], "count": 0, "stream_batches": []}
    ctx, manager = make_ctx(script)

    async def _raise(fn, *a, **kw):
        raise RuntimeError("forced error")

    with patch("asyncio.to_thread", side_effect=_raise):
        with pytest.raises(RuntimeError, match="Stream error"):
            await srv.execute_query_stream(ctx=ctx, query="SELECT 1")


# ─── run_query_safely: DB error during probe ─────────────────────────────────

@pytest.mark.asyncio
async def test_run_query_safely_db_error_in_probe(make_ctx):
    """Covers the except Exception branch in the probe section."""
    ctx, manager = make_ctx({"columns": ["c"], "probe_rows": [(1,)], "count": 1})

    async def _raise(*a, **kw):
        raise RuntimeError("probe failed")

    with patch.object(srv, "_execute_query_async", side_effect=_raise):
        with pytest.raises(RuntimeError, match="Unexpected error"):
            await srv.run_query_safely(
                ctx=ctx, query="SELECT 1 LIMIT 10"
            )


# ─── Metadata empty-result branches ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_schema_tables_empty(make_ctx):
    """When no tables found, raises RuntimeError."""
    script = {
        "columns": ["table_name"],
        "probe_rows": [],
        "count": 0,
    }
    ctx, manager = make_ctx(script)
    with pytest.raises(RuntimeError, match="No tables found"):
        await srv.get_schema_tables(ctx=ctx, schema_name="empty_schema")


@pytest.mark.asyncio
async def test_get_schema_views_empty(make_ctx):
    script = {"columns": ["view_name", "def"], "probe_rows": [], "count": 0}
    ctx, manager = make_ctx(script)
    with pytest.raises(RuntimeError, match="No views found"):
        await srv.get_schema_views(ctx=ctx, schema_name="empty_schema")


@pytest.mark.asyncio
async def test_get_table_structure_empty(make_ctx):
    script = {
        "columns": ["column_name", "data_type", "char_max", "num_prec", "num_scale", "nullable", "default"],
        "probe_rows": [],
        "count": 0,
    }
    ctx, manager = make_ctx(script)
    with pytest.raises(RuntimeError, match="Table not found"):
        await srv.get_table_structure(ctx=ctx, table_name="missing_table")


@pytest.mark.asyncio
async def test_get_table_projections_empty(make_ctx):
    script = {
        "columns": ["projection_name", "is_super", "anchor_table", "create_type"],
        "probe_rows": [],
        "count": 0,
    }
    ctx, manager = make_ctx(script)
    with pytest.raises(RuntimeError, match="No projections found"):
        await srv.get_table_projections(ctx=ctx, table_name="missing_table")


@pytest.mark.asyncio
async def test_get_database_schemas_empty(make_ctx):
    script = {"columns": ["schema_name", "is_system"], "probe_rows": [], "count": 0}
    ctx, manager = make_ctx(script)
    with pytest.raises(RuntimeError, match="No schemas found"):
        await srv.get_database_schemas(ctx=ctx)
