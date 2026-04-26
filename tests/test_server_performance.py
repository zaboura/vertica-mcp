"""Tests for database performance and status MCP tools in server.py."""
import pytest
from unittest.mock import patch, AsyncMock

from tests.conftest import make_ctx
import vertica_mcp.server as server_module


# ─── Fixtures ────────────────────────────────────────────────────────────────

FAKE_STATUS = {
    "result": "Database Status Report\nVersion: Vertica 24.x\n",
    "version": "Vertica 24.x",
    "current_usage_pct": 45.0,
    "current_db_size_gb": 90.0,
    "license_capacity_gb": 200.0,
    "trend_data_points": 3,
    "cluster_nodes": 3,
}

FAKE_PERF = {
    "cpu": [{"node_name": "n1", "ts": "2024-01-01", "cpu_pct": 40.0, "mem_pct": 50.0}],
    "memory": [{"node_name": "n1", "ts": "2024-01-01", "cpu_pct": 40.0, "mem_pct": 50.0}],
    "top_tables_by_ros": [{"anchor_table_name": "fact_sales", "total_ros_containers": 100}],
    "meta": {"window_minutes": 5, "bucket": "minute", "top_n": 3},
}

FAKE_PROFILE = {
    "result": "Execution Time: 1000μs\nTransaction ID: 1\nStatement ID: 1\n\nExecution Plan:\nPath",
    "query": "SELECT 1",
    "label": "mcp_profile_abc123",
    "transaction_id": "1",
    "statement_id": "1",
    "duration_us": 1000,
    "plan_line_count": 1,
}


@pytest.fixture
def perf_script():
    return {
        "columns": ["node_name", "ts", "cpu_pct", "mem_pct", "col5", "col6"],
        "probe_rows": [
            ("n1", "2024-01-01", 40.0, 50.0, "a", "b"),
            ("n2", "2024-01-01", 60.0, 70.0, "c", "d"),
        ],
        "count": 2,
    }


# ─── database_status ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_database_status_success(make_ctx, perf_script):
    ctx, manager = make_ctx(perf_script)
    with patch("asyncio.to_thread", return_value=FAKE_STATUS):
        result = await server_module.database_status(ctx=ctx)

    assert isinstance(result, dict)
    assert "version" in result
    assert "cluster_nodes" in result
    assert result["version"] == "Vertica 24.x"
    assert result["cluster_nodes"] == 3


@pytest.mark.asyncio
async def test_database_status_no_connection(make_ctx, perf_script):
    ctx, _ = make_ctx(perf_script)
    ctx.request_context.lifespan_context.clear()
    with pytest.raises(RuntimeError, match="No database connection manager available"):
        await server_module.database_status(ctx=ctx)


# ─── profile_query ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_profile_query_success(make_ctx, perf_script):
    ctx, manager = make_ctx(perf_script)
    with patch("asyncio.to_thread", return_value=FAKE_PROFILE):
        result = await server_module.profile_query(
            ctx=ctx, query="SELECT 1", timeout=5
        )

    assert isinstance(result, dict)
    assert "result" in result
    assert "transaction_id" in result
    assert result["transaction_id"] == "1"


@pytest.mark.asyncio
async def test_profile_query_no_connection(make_ctx, perf_script):
    ctx, _ = make_ctx(perf_script)
    ctx.request_context.lifespan_context.clear()
    with pytest.raises(RuntimeError, match="No database connection manager available"):
        await server_module.profile_query(ctx=ctx, query="SELECT 1")


# ─── analyze_system_performance ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analyze_system_performance_success(make_ctx, perf_script):
    ctx, manager = make_ctx(perf_script)
    with patch("asyncio.to_thread", return_value=FAKE_PERF):
        result = await server_module.analyze_system_performance(ctx=ctx, window_minutes=5)

    assert isinstance(result, dict)
    assert "cpu" in result
    assert "memory" in result
    assert "top_tables_by_ros" in result
    assert "meta" in result


@pytest.mark.asyncio
async def test_analyze_system_performance_no_connection(make_ctx, perf_script):
    ctx, _ = make_ctx(perf_script)
    ctx.request_context.lifespan_context.clear()
    with pytest.raises(RuntimeError, match="No database connection manager available"):
        await server_module.analyze_system_performance(ctx=ctx)


# ─── generate_health_dashboard ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_health_dashboard_compact(make_ctx, perf_script):
    ctx, manager = make_ctx(perf_script)
    # Patch the two async calls it depends on
    with (
        patch.object(server_module, "database_status", new=AsyncMock(return_value=FAKE_STATUS)),
        patch.object(server_module, "analyze_system_performance", new=AsyncMock(return_value=FAKE_PERF)),
    ):
        result = await server_module.generate_health_dashboard(
            ctx=ctx, output_format="compact"
        )

    assert isinstance(result, dict)
    assert "result" in result
    assert "token_estimate" in result


@pytest.mark.asyncio
async def test_generate_health_dashboard_detailed(make_ctx, perf_script):
    ctx, manager = make_ctx(perf_script)
    with (
        patch.object(server_module, "database_status", new=AsyncMock(return_value=FAKE_STATUS)),
        patch.object(server_module, "analyze_system_performance", new=AsyncMock(return_value=FAKE_PERF)),
    ):
        result = await server_module.generate_health_dashboard(
            ctx=ctx, output_format="detailed"
        )

    assert isinstance(result, dict)
    assert "result" in result
    assert "token_estimate" in result


@pytest.mark.asyncio
async def test_generate_health_dashboard_json(make_ctx, perf_script):
    ctx, manager = make_ctx(perf_script)
    with (
        patch.object(server_module, "database_status", new=AsyncMock(return_value=FAKE_STATUS)),
        patch.object(server_module, "analyze_system_performance", new=AsyncMock(return_value=FAKE_PERF)),
    ):
        result = await server_module.generate_health_dashboard(
            ctx=ctx, output_format="json"
        )

    assert isinstance(result, dict)
    assert "version" in result
    assert "metrics" in result
    assert "alerts" in result
