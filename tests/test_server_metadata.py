"""Tests for database metadata MCP tools in server.py."""
import pytest
from unittest.mock import patch, AsyncMock
from tests.conftest import make_ctx
import vertica_mcp.server as server_module

# Script providing 7 columns to satisfy get_table_structure's 7-col column query
# (column_name, data_type, char_max_len, numeric_precision, numeric_scale, is_nullable, column_default)
@pytest.fixture
def metadata_script():
    return {
        "columns": ["column_name", "data_type", "character_maximum_length",
                    "numeric_precision", "numeric_scale", "is_nullable", "column_default"],
        "probe_rows": [
            ("id", "int", None, 10, 0, "NO", None),
            ("name", "varchar", 100, None, None, "YES", None),
        ],
        "count": 2,
    }


@pytest.mark.asyncio
async def test_get_database_schemas(make_ctx, metadata_script):
    ctx, manager = make_ctx(metadata_script)
    result = await server_module.get_database_schemas(ctx=ctx)

    assert isinstance(result, dict)
    assert "result" in result
    assert "schema_count" in result
    assert result["schema_count"] == 2


@pytest.mark.asyncio
async def test_get_schema_tables(make_ctx, metadata_script):
    ctx, manager = make_ctx(metadata_script)
    result = await server_module.get_schema_tables(ctx=ctx, schema_name="public")

    assert isinstance(result, dict)
    assert "result" in result
    assert "table_count" in result
    assert result["table_count"] == 2
    assert result["schema"] == "public"


@pytest.mark.asyncio
async def test_get_schema_views(make_ctx, metadata_script):
    ctx, manager = make_ctx(metadata_script)
    result = await server_module.get_schema_views(ctx=ctx, schema_name="public")

    assert isinstance(result, dict)
    assert "result" in result
    assert "view_count" in result
    assert result["view_count"] == 2


@pytest.mark.asyncio
async def test_get_table_structure(make_ctx, metadata_script):
    ctx, manager = make_ctx(metadata_script)
    result = await server_module.get_table_structure(
        ctx=ctx, table_name="test_table", schema_name="public"
    )

    assert isinstance(result, dict)
    assert "result" in result
    assert "column_count" in result
    assert result["column_count"] == 2
    assert result["table_name"] == "test_table"


@pytest.mark.asyncio
async def test_get_table_projections(make_ctx, metadata_script):
    ctx, manager = make_ctx(metadata_script)
    result = await server_module.get_table_projections(
        ctx=ctx, table_name="test_table", schema_name="public"
    )

    assert isinstance(result, dict)
    assert "result" in result
    assert "projection_count" in result
    assert result["projection_count"] == 2


@pytest.mark.asyncio
async def test_get_database_schemas_no_connection(make_ctx, metadata_script):
    """Ensure error is raised when no manager is available."""
    ctx, manager = make_ctx(metadata_script)
    # Remove the manager from context
    ctx.request_context.lifespan_context.clear()
    with pytest.raises(RuntimeError, match="No database connection manager available"):
        await server_module.get_database_schemas(ctx=ctx)


@pytest.mark.asyncio
async def test_get_schema_tables_no_connection(make_ctx, metadata_script):
    ctx, manager = make_ctx(metadata_script)
    ctx.request_context.lifespan_context.clear()
    with pytest.raises(RuntimeError, match="No database connection manager available"):
        await server_module.get_schema_tables(ctx=ctx, schema_name="public")
