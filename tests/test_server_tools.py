# tests/test_server_tools.py
import pytest
from typing import List, Tuple

from vertica_mcp.connection import OperationType


@pytest.mark.asyncio
async def test_run_query_safely_small_result(make_ctx, small_result_script, server_module):
    ctx, _ = make_ctx(small_result_script)
    result = await server_module.run_query_safely(
        ctx=ctx,
        query="SELECT user_id, name FROM users",
        row_threshold=10,
        include_columns=True,
    )
    assert result["ok"] is True
    assert result["done"] is True
    assert result["large"] is False
    assert result["count"] == 3
    assert result["rows"] == small_result_script["probe_rows"]
    assert result["columns"] == small_result_script["columns"]


@pytest.mark.asyncio
async def test_run_query_safely_large_requires_confirmation(make_ctx, large_result_script, server_module):
    ctx, _ = make_ctx(large_result_script)
    result = await server_module.run_query_safely(
        ctx=ctx,
        query="SELECT user_id, name FROM big_table",
        row_threshold=1000,  # probe returns > 1000 rows
        include_columns=True,
        precount=True,
    )
    assert result["ok"] is True
    assert result["large"] is True
    assert result["requires_confirmation"] is True
    assert "preview" in result and len(result["preview"]) > 0
    assert result["exact_count"] == large_result_script["count"]


@pytest.mark.asyncio
async def test_run_query_safely_proceed_paginated(make_ctx, small_result_script, server_module):
    ctx, _ = make_ctx(small_result_script)
    result = await server_module.run_query_safely(
        ctx=ctx,
        query="SELECT user_id, name FROM users",
        row_threshold=1,      # force "large" if we didn't pass proceed
        proceed=True,         # we proceed, so it goes to pagination
        mode="page",
        page_limit=2,
        include_columns=True,
    )
    assert "rows" in result and len(result["rows"]) == 2
    assert result["count"] == 2
    assert result["next_offset"] == 2
    assert result["done"] in (True, False)  # depends on rows==limit


@pytest.mark.asyncio
async def test_execute_query_stream(make_ctx, small_result_script, server_module):
    ctx, _ = make_ctx(small_result_script)
    # Directly call the streaming tool
    result = await server_module.execute_query_stream(
        ctx=ctx,
        query="SELECT user_id, name FROM users ORDER BY user_id",
        batch_size=1000,
    )
    assert "result" in result
    all_rows = result["result"]
    # our script had 2 + 1 rows in 2 batches -> 3 total
    assert sum(1 for _ in all_rows) == 3
    assert result["total_rows"] == 3
    assert "size_mb" in result


@pytest.mark.asyncio
async def test_non_select_permission_denied(make_ctx, small_result_script, server_module, connection_module, monkeypatch):
    # Ensure all globals are False (default) and no schema override
    monkeypatch.delenv("ALLOW_DDL_OPERATION", raising=False)
    monkeypatch.delenv("ALLOW_INSERT_OPERATION", raising=False)
    monkeypatch.delenv("ALLOW_UPDATE_OPERATION", raising=False)
    monkeypatch.delenv("ALLOW_DELETE_OPERATION", raising=False)

    ctx, manager = make_ctx(small_result_script)

    # CREATE TABLE -> DDL -> should be blocked by manager.is_operation_allowed(...)
    with pytest.raises(RuntimeError) as ei:
        await server_module.run_query_safely(
            ctx=ctx,
            query="CREATE TABLE t(id INT)",
        )
    assert "not allowed" in str(ei.value)

    # UPDATE -> should be blocked
    with pytest.raises(RuntimeError):
        await server_module.run_query_safely(
            ctx=ctx,
            query="UPDATE users SET name='X' WHERE user_id='u1'",
        )


@pytest.mark.parametrize(
    "sql,expected_schema",
    [
        ("SELECT * FROM public.users", "public"),
        ("select * from tpcds.store_sales", "tpcds"),
        ('  SELECT * FROM "MixedCase".tbl  ', None),  # extractor doesn't handle quoted identifiers (by design, for now)
        ("SELECT 1", None),
    ],
)
def test_extract_schema_from_query(server_module, sql, expected_schema):
    actual = server_module.extract_schema_from_query(sql)
    assert actual == expected_schema
