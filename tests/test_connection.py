# tests/test_connection.py
import os
import pytest
from vertica_mcp.connection import VerticaConfig, VerticaConnectionManager, SchemaPermissions


def test_config_from_env_parses_booleans_and_schema_perms(monkeypatch):
    monkeypatch.setenv("VERTICA_HOST", "10.0.0.1")
    monkeypatch.setenv("VERTICA_PORT", "5433")
    monkeypatch.setenv("VERTICA_DATABASE", "VMart")
    monkeypatch.setenv("VERTICA_USER", "dbadmin")
    monkeypatch.setenv("VERTICA_PASSWORD", "pwd")
    monkeypatch.setenv("VERTICA_CONNECTION_LIMIT", "7")
    monkeypatch.setenv("VERTICA_SSL", "true")
    monkeypatch.setenv("VERTICA_SSL_REJECT_UNAUTHORIZED", "false")
    monkeypatch.setenv("ALLOW_INSERT_OPERATION", "true")
    monkeypatch.setenv("ALLOW_UPDATE_OPERATION", "false")
    monkeypatch.setenv("ALLOW_DELETE_OPERATION", "false")
    monkeypatch.setenv("ALLOW_DDL_OPERATION", "false")
    monkeypatch.setenv("SCHEMA_INSERT_PERMISSIONS", "public:true,tpcds:false")
    monkeypatch.setenv("SCHEMA_UPDATE_PERMISSIONS", "public:false")
    monkeypatch.setenv("SCHEMA_DELETE_PERMISSIONS", "public:false")
    monkeypatch.setenv("SCHEMA_DDL_PERMISSIONS", "public:false,tpcds:false")

    cfg = VerticaConfig.from_env()
    assert cfg.host == "10.0.0.1"
    assert cfg.port == 5433
    assert cfg.database == "VMart"
    assert cfg.user == "dbadmin"
    assert cfg.password == "pwd"
    assert cfg.connection_limit == 7
    assert cfg.ssl is True
    assert cfg.ssl_reject_unauthorized is False
    # global
    assert cfg.allow_insert is True
    assert cfg.allow_update is False
    # per-schema
    perms = cfg.schema_permissions or {}
    assert isinstance(perms.get("public"), SchemaPermissions)
    assert perms["public"].insert is True
    assert perms["public"].update is False
    assert perms["public"].delete is False
    assert perms["public"].ddl is False
    assert perms["tpcds"].insert is False


def test_is_operation_allowed_prefers_schema_over_global(monkeypatch):
    # Global: insert = False; Schema public: insert=True
    monkeypatch.setenv("ALLOW_INSERT_OPERATION", "false")
    monkeypatch.setenv("SCHEMA_INSERT_PERMISSIONS", "public:true")
    monkeypatch.setenv("VERTICA_DATABASE", "VMart")
    monkeypatch.setenv("VERTICA_LAZY_INIT", "1")

    cfg = VerticaConfig.from_env()
    mgr = VerticaConnectionManager()
    mgr.initialize_default(cfg)

    from vertica_mcp.connection import OperationType
    assert mgr.is_operation_allowed("public", OperationType.INSERT) is True
    assert mgr.is_operation_allowed("public", OperationType.DDL) is False

def test_oauth_requires_token():
    from vertica_mcp.connection import VerticaConnectionPool
    config = VerticaConfig(host="localhost", port=5433, database="db", user="u", password="", auth_mode="oauth", oauth_token="")
    try:
        pool = VerticaConnectionPool(config)
        pool._get_connection_config()
        assert False, "Should have raised ValueError for missing oauth token"
    except ValueError as e:
        assert "oauth_token is required" in str(e)

def test_mtls_requires_certs():
    from vertica_mcp.connection import VerticaConnectionPool
    config = VerticaConfig(host="localhost", port=5433, database="db", user="u", password="", auth_mode="mtls", ssl_cert="", ssl_key="")
    try:
        pool = VerticaConnectionPool(config)
        pool._get_connection_config()
        assert False, "Should have raised ValueError for missing certs"
    except ValueError as e:
        assert "ssl_cert and ssl_key are required" in str(e)
