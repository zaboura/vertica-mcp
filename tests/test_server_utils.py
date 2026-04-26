"""Tests for pure utility functions in server.py — highest coverage gain per test."""
import time
import pytest

import vertica_mcp.server as srv

# ─── _strip_sql_comments ─────────────────────────────────────────────────────

def test_strip_sql_comments_block():
    assert srv._strip_sql_comments("SELECT /* comment */ 1").strip() == "SELECT  1"

def test_strip_sql_comments_line():
    result = srv._strip_sql_comments("SELECT 1 -- trailing comment\n")
    assert "trailing" not in result

def test_strip_sql_comments_clean():
    q = "SELECT col FROM t"
    assert srv._strip_sql_comments(q) == q


# ─── _is_select ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("q", [
    "SELECT 1",
    "select col from t",
    "WITH cte AS (SELECT 1) SELECT * FROM cte",
    "EXPLAIN SELECT 1",
    "PROFILE SELECT 1",
    "(SELECT 1)",
])
def test_is_select_true(q):
    assert srv._is_select(q) is True

@pytest.mark.parametrize("q", [
    "INSERT INTO t VALUES (1)",
    "UPDATE t SET c=1",
    "DELETE FROM t",
    "DROP TABLE t",
    "",
])
def test_is_select_false(q):
    assert srv._is_select(q) is False


# ─── _wrap_subquery ───────────────────────────────────────────────────────────

def test_wrap_subquery_basic():
    wrapped = srv._wrap_subquery("SELECT 1 FROM t;")
    assert wrapped == "SELECT * FROM (SELECT 1 FROM t) q"

def test_wrap_subquery_strips_semicolon():
    result = srv._wrap_subquery("SELECT a FROM b;")
    assert ";" not in result


# ─── _sanitize_query ─────────────────────────────────────────────────────────

def test_sanitize_query_valid_select():
    q = "SELECT id FROM users LIMIT 10"
    assert srv._sanitize_query(q) == q

def test_sanitize_query_valid_with():
    q = "WITH cte AS (SELECT 1) SELECT * FROM cte"
    assert srv._sanitize_query(q) == q

def test_sanitize_query_valid_explain():
    q = "EXPLAIN SELECT 1"
    assert srv._sanitize_query(q) == q

def test_sanitize_query_valid_show():
    assert srv._sanitize_query("SHOW TABLES") == "SHOW TABLES"

def test_sanitize_query_rejects_insert():
    with pytest.raises(ValueError, match="Only read-only"):
        srv._sanitize_query("INSERT INTO t VALUES (1)")

def test_sanitize_query_rejects_delete():
    with pytest.raises(ValueError, match="Only read-only"):
        srv._sanitize_query("DELETE FROM t")

def test_sanitize_query_rejects_update():
    with pytest.raises(ValueError, match="Only read-only"):
        srv._sanitize_query("UPDATE t SET c=1")

def test_sanitize_query_rejects_drop():
    with pytest.raises(ValueError, match="Only read-only"):
        srv._sanitize_query("DROP TABLE t")

def test_sanitize_query_empty():
    with pytest.raises(ValueError, match="non-empty"):
        srv._sanitize_query("")

def test_sanitize_query_none():
    with pytest.raises(ValueError, match="non-empty"):
        srv._sanitize_query(None)

def test_sanitize_query_too_long():
    with pytest.raises(ValueError, match="too long"):
        srv._sanitize_query("SELECT " + "a" * 50001)

def test_sanitize_query_blocks_union_injection():
    with pytest.raises(ValueError, match="UNION"):
        srv._sanitize_query("SELECT 1 UNION ALL SELECT 2")

def test_sanitize_query_blocks_sleep():
    with pytest.raises(ValueError, match="Time-based"):
        srv._sanitize_query("SELECT SLEEP(5)")

def test_sanitize_query_blocks_waitfor():
    with pytest.raises(ValueError, match="Time-based"):
        srv._sanitize_query("SELECT WAITFOR(1)")


# ─── _check_rate_limit ───────────────────────────────────────────────────────

def test_rate_limit_allows_first_call():
    # Use unique client id to avoid cross-test pollution
    client_id = f"test-client-{time.time()}"
    assert srv._check_rate_limit(client_id) is True

def test_rate_limit_blocks_over_limit():
    client_id = f"test-rl-over-{time.time()}"
    # Saturate the tracker manually
    srv.rate_limit_tracker[client_id] = [time.time()] * srv.RATE_LIMIT_PER_MINUTE
    assert srv._check_rate_limit(client_id) is False

def test_rate_limit_cleans_old_entries():
    client_id = f"test-rl-old-{time.time()}"
    # Insert old timestamps (> 60 s ago)
    old_ts = time.time() - 120
    srv.rate_limit_tracker[client_id] = [old_ts] * srv.RATE_LIMIT_PER_MINUTE
    assert srv._check_rate_limit(client_id) is True


# ─── _get_cached_metadata / _set_cached_metadata ────────────────────────────

def test_cache_miss():
    srv._get_cached_metadata.cache_clear()
    srv.metadata_cache.clear()
    result = srv._get_cached_metadata("nonexistent_key_xyz")
    assert result is None

def test_cache_hit():
    srv._get_cached_metadata.cache_clear()
    srv.metadata_cache.clear()
    key = "test_cache_key"
    srv._set_cached_metadata(key, {"data": "value"})
    srv._get_cached_metadata.cache_clear()   # clear lru so it reads fresh
    result = srv._get_cached_metadata(key)
    assert result == {"data": "value"}

def test_cache_expired():
    srv._get_cached_metadata.cache_clear()
    srv.metadata_cache.clear()
    key = "test_expired_key"
    # Set with old timestamp
    srv.metadata_cache[key] = ({"old": True}, time.time() - srv.CACHE_TTL_SECONDS - 1)
    srv._get_cached_metadata.cache_clear()
    result = srv._get_cached_metadata(key)
    assert result is None
    assert key not in srv.metadata_cache


# ─── extract_operation_type ──────────────────────────────────────────────────

def test_extract_operation_type_insert():
    from vertica_mcp.server import OperationType
    assert srv.extract_operation_type("INSERT INTO t VALUES (1)") == OperationType.INSERT

def test_extract_operation_type_update():
    from vertica_mcp.server import OperationType
    assert srv.extract_operation_type("UPDATE t SET c=1") == OperationType.UPDATE

def test_extract_operation_type_delete():
    from vertica_mcp.server import OperationType
    assert srv.extract_operation_type("DELETE FROM t") == OperationType.DELETE

def test_extract_operation_type_ddl_create():
    from vertica_mcp.server import OperationType
    assert srv.extract_operation_type("CREATE TABLE t (id INT)") == OperationType.DDL

def test_extract_operation_type_ddl_alter():
    from vertica_mcp.server import OperationType
    assert srv.extract_operation_type("ALTER TABLE t ADD COLUMN c INT") == OperationType.DDL

def test_extract_operation_type_ddl_drop():
    from vertica_mcp.server import OperationType
    assert srv.extract_operation_type("DROP TABLE t") == OperationType.DDL

def test_extract_operation_type_ddl_truncate():
    from vertica_mcp.server import OperationType
    assert srv.extract_operation_type("TRUNCATE TABLE t") == OperationType.DDL

def test_extract_operation_type_select_returns_none():
    assert srv.extract_operation_type("SELECT 1") is None


# ─── extract_schema_from_query ───────────────────────────────────────────────

def test_extract_schema_found():
    result = srv.extract_schema_from_query("SELECT * FROM public.users")
    assert result == "public"

def test_extract_schema_not_found():
    result = srv.extract_schema_from_query("SELECT 1")
    assert result is None


# ─── _inject_label ───────────────────────────────────────────────────────────

def test_inject_label_simple_select():
    result = srv._inject_label("SELECT 1", "my_label")
    assert "/*+LABEL('my_label')*/" in result
    assert result.startswith("SELECT")

def test_inject_label_uppercase():
    result = srv._inject_label("SELECT col FROM t", "lbl")
    assert "/*+LABEL('lbl')*/" in result

def test_inject_label_with_cte():
    sql = "WITH cte AS (SELECT 1) SELECT * FROM cte"
    result = srv._inject_label(sql, "cte_label")
    assert "/*+LABEL('cte_label')*/" in result

def test_inject_label_strips_existing():
    sql = "SELECT /*+LABEL('old')*/ 1"
    result = srv._inject_label(sql, "new_label")
    assert "old" not in result
    assert "new_label" in result

def test_inject_label_with_string_literals():
    sql = "SELECT 'select me' FROM t"
    result = srv._inject_label(sql, "str_lbl")
    assert "/*+LABEL('str_lbl')*/" in result

def test_inject_label_subquery():
    sql = "SELECT * FROM (SELECT id FROM users) sub"
    result = srv._inject_label(sql, "sub_label")
    # Label should be injected at the outer SELECT
    assert "/*+LABEL('sub_label')*/" in result

def test_inject_label_with_line_comment():
    sql = "-- comment\nSELECT 1"
    result = srv._inject_label(sql, "lbl")
    assert "/*+LABEL('lbl')*/" in result

def test_inject_label_with_block_comment():
    sql = "/* block */ SELECT 1"
    result = srv._inject_label(sql, "blk_lbl")
    assert "/*+LABEL('blk_lbl')*/" in result

def test_inject_label_no_select_falls_back():
    # Non-SELECT gets the prepended fallback
    sql = "EXPLAIN QUERY PLAN"
    result = srv._inject_label(sql, "fallback_lbl")
    assert "/*+LABEL('fallback_lbl')*/" in result

def test_inject_label_double_quoted_identifier():
    sql = 'SELECT "select" FROM t'
    result = srv._inject_label(sql, "dq_lbl")
    assert "/*+LABEL('dq_lbl')*/" in result

def test_inject_label_single_quoted_select_inside():
    sql = "SELECT 'SELECT inside string' FROM t"
    result = srv._inject_label(sql, "sq_lbl")
    assert "/*+LABEL('sq_lbl')*/" in result


# ─── _calculate_avg ──────────────────────────────────────────────────────────

def test_calculate_avg_normal():
    data = [{"cpu_pct": 10.0}, {"cpu_pct": 20.0}, {"cpu_pct": 30.0}]
    assert srv._calculate_avg(data, "cpu_pct") == pytest.approx(20.0)

def test_calculate_avg_empty():
    assert srv._calculate_avg([], "cpu_pct") == 0.0

def test_calculate_avg_missing_field():
    data = [{"other": 5}, {"other": 10}]
    assert srv._calculate_avg(data, "cpu_pct") == 0.0

def test_calculate_avg_mixed():
    data = [{"cpu_pct": 50.0}, {"other": 10}]
    assert srv._calculate_avg(data, "cpu_pct") == pytest.approx(50.0)


# ─── _generate_alerts ────────────────────────────────────────────────────────

def test_generate_alerts_no_alerts():
    perf = {
        "cpu": [{"cpu_pct": 30.0}],
        "memory": [{"mem_pct": 40.0}],
        "top_tables_by_ros": [{"anchor_table_name": "t", "total_ros_containers": 100}],
    }
    alerts = srv._generate_alerts(perf)
    assert alerts == []

def test_generate_alerts_high_cpu():
    perf = {
        "cpu": [{"cpu_pct": 90.0}],
        "memory": [{"mem_pct": 40.0}],
        "top_tables_by_ros": [],
    }
    alerts = srv._generate_alerts(perf)
    types = [a["type"] for a in alerts]
    assert "cpu_high" in types

def test_generate_alerts_high_memory():
    perf = {
        "cpu": [{"cpu_pct": 30.0}],
        "memory": [{"mem_pct": 92.0}],
        "top_tables_by_ros": [],
    }
    alerts = srv._generate_alerts(perf)
    types = [a["type"] for a in alerts]
    assert "memory_high" in types

def test_generate_alerts_high_ros():
    perf = {
        "cpu": [],
        "memory": [],
        "top_tables_by_ros": [
            {"anchor_table_name": "big_table", "total_ros_containers": 6000}
        ],
    }
    alerts = srv._generate_alerts(perf)
    types = [a["type"] for a in alerts]
    assert "ros_high" in types

def test_generate_alerts_empty_perf():
    alerts = srv._generate_alerts({})
    assert alerts == []


# ─── _format_compact_dashboard ───────────────────────────────────────────────

def test_format_compact_dashboard():
    status = {"version": "Vertica 24.x"}
    perf = {
        "cpu": [{"cpu_pct": 40.0}],
        "memory": [{"mem_pct": 50.0}],
        "top_tables_by_ros": [],
    }
    result = srv._format_compact_dashboard(status, perf)
    assert "CPU" in result
    assert "Mem" in result

def test_format_compact_dashboard_with_alerts():
    status = {"version": "Vertica 24.x"}
    perf = {
        "cpu": [{"cpu_pct": 95.0}],
        "memory": [{"mem_pct": 90.0}],
        "top_tables_by_ros": [],
    }
    result = srv._format_compact_dashboard(status, perf)
    assert "alert" in result.lower()


# ─── _format_detailed_dashboard ──────────────────────────────────────────────

def test_format_detailed_dashboard():
    status = {"version": "Vertica 24.x"}
    perf = {
        "cpu": [{"cpu_pct": 40.0}],
        "memory": [{"mem_pct": 50.0}],
        "top_tables_by_ros": [
            {"anchor_table_name": "fact_sales", "total_ros_containers": 100}
        ],
        "meta": {"window_minutes": 5},
    }
    result = srv._format_detailed_dashboard(status, perf)
    assert "Version" in result
    assert "CPU" in result
    assert "Memory" in result

def test_format_detailed_dashboard_no_tables():
    status = {"version": "Vertica"}
    perf = {"cpu": [], "memory": [], "top_tables_by_ros": []}
    result = srv._format_detailed_dashboard(status, perf)
    assert "No data" in result or "Alerts" in result

def test_format_detailed_dashboard_with_alerts():
    status = {"version": "Vertica"}
    perf = {
        "cpu": [{"cpu_pct": 95.0}],
        "memory": [{"mem_pct": 92.0}],
        "top_tables_by_ros": [],
    }
    result = srv._format_detailed_dashboard(status, perf)
    assert "Alerts" in result
