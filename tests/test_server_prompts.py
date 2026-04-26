import pytest
from unittest.mock import patch

import vertica_mcp.server as server_module

@pytest.mark.asyncio
async def test_vertica_database_health_dashboard():
    result = await server_module.vertica_database_health_dashboard()
    assert "HEALTH DASHBOARD" in result

@pytest.mark.asyncio
async def test_vertica_database_system_monitor():
    result = await server_module.vertica_database_system_monitor()
    assert "SYSTEM MONITOR" in result

@pytest.mark.asyncio
async def test_vertica_compact_health_report():
    result = await server_module.vertica_compact_health_report()
    assert "COMPACT REPORT" in result

@pytest.mark.asyncio
async def test_sql_query_safety_guard():
    result = await server_module.sql_query_safety_guard()
    assert "SQL SAFETY" in result

@pytest.mark.asyncio
async def test_vertica_query_performance_analyzer():
    result = await server_module.vertica_query_performance_analyzer()
    assert "VERTICA PERFORMANCE ANALYSIS" in result

@pytest.mark.asyncio
async def test_vertica_sql_assistant():
    result = await server_module.vertica_sql_assistant()
    assert "VERTICA SQL QUERY GENERATION" in result
