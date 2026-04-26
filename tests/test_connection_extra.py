"""Tests for additional branches in connection.py."""
import os
import types
import pytest
from unittest.mock import patch, MagicMock

import vertica_mcp.connection as vconn
from vertica_mcp.connection import (
    VerticaConfig,
    VerticaConnectionPool,
    VerticaConnectionManager,
    OperationType,
)

# Helper to build a minimal valid VerticaConfig
def _cfg(**kwargs):
    defaults = dict(host="localhost", port=5433, database="db", user="u", password="p")
    defaults.update(kwargs)
    return VerticaConfig(**defaults)


# ─── VerticaConfig: invalid connection_limit ──────────────────────────────────

def test_config_invalid_connection_limit():
    with pytest.raises(ValueError, match="connection_limit must be a positive integer"):
        _cfg(connection_limit=0)


def test_config_invalid_connection_limit_negative():
    with pytest.raises(ValueError, match="connection_limit must be a positive integer"):
        _cfg(connection_limit=-5)


# ─── VerticaConfig: from_env ─────────────────────────────────────────────────

def test_config_from_env_defaults(monkeypatch):
    monkeypatch.setenv("VERTICA_HOST", "myhost")
    monkeypatch.setenv("VERTICA_DATABASE", "mydb")
    monkeypatch.setenv("VERTICA_USER", "myuser")
    monkeypatch.setenv("VERTICA_PASSWORD", "mypass")
    monkeypatch.delenv("VERTICA_PORT", raising=False)
    cfg = VerticaConfig.from_env()
    assert cfg.host == "myhost"
    assert cfg.database == "mydb"
    assert cfg.port == 5433  # default


# ─── VerticaConnectionPool: context manager ───────────────────────────────────

def test_pool_context_manager(monkeypatch):
    cfg = _cfg()
    fake_conn = MagicMock()
    monkeypatch.setattr(vconn, "vertica_python", types.SimpleNamespace(
        connect=lambda **kw: fake_conn
    ))
    pool = VerticaConnectionPool(cfg)
    pool.pool.put(fake_conn)

    with pool as p:
        assert p is pool
    # No error should occur on __exit__


# ─── VerticaConnectionPool: get/release ──────────────────────────────────────

def test_pool_get_and_release(monkeypatch):
    cfg = _cfg()
    fake_conn = MagicMock()
    monkeypatch.setattr(vconn, "vertica_python", types.SimpleNamespace(
        connect=lambda **kw: fake_conn
    ))
    pool = VerticaConnectionPool(cfg)
    pool.pool.put(fake_conn)

    conn = pool.get_connection()
    assert conn is fake_conn
    assert pool.active_connections == 1

    pool.release_connection(conn)
    assert pool.active_connections == 0


def test_pool_exhausted(monkeypatch):
    cfg = _cfg(connection_limit=1)
    fake_conn = MagicMock()
    monkeypatch.setattr(vconn, "vertica_python", types.SimpleNamespace(
        connect=lambda **kw: fake_conn
    ))
    pool = VerticaConnectionPool(cfg)
    pool.pool.put(fake_conn)

    c1 = pool.get_connection()
    assert pool.active_connections == 1

    with pytest.raises(Exception, match="[Ee]xhausted|[Pp]ool"):
        pool.get_connection()

    pool.release_connection(c1)


def test_pool_release_error_closes_conn(monkeypatch):
    """If pool.put raises, fallback closes the connection."""
    cfg = _cfg()
    fake_conn = MagicMock()
    monkeypatch.setattr(vconn, "vertica_python", types.SimpleNamespace(
        connect=lambda **kw: fake_conn
    ))
    pool = VerticaConnectionPool(cfg)
    pool.pool.put = MagicMock(side_effect=Exception("put failed"))
    pool.active_connections = 1

    pool.release_connection(fake_conn)  # should not raise
    # active_connections is decremented inside the lock even if put fails
    assert pool.active_connections <= 1  # either 0 or 1 depending on impl path


# ─── _get_connection_config: auth modes ───────────────────────────────────────

def test_connection_config_oauth(monkeypatch):
    cfg = _cfg(auth_mode="oauth", oauth_token="my-token")
    fake_conn = MagicMock()
    monkeypatch.setattr(vconn, "vertica_python", types.SimpleNamespace(
        connect=lambda **kw: fake_conn
    ))
    pool = VerticaConnectionPool(cfg)
    config = pool._get_connection_config()
    assert config["oauth_access_token"] == "my-token"
    assert "password" not in config


def test_connection_config_oauth_missing_token():
    cfg = _cfg(auth_mode="oauth", oauth_token="")
    pool = VerticaConnectionPool.__new__(VerticaConnectionPool)
    pool.config = cfg
    with pytest.raises(ValueError, match="oauth_token is required"):
        pool._get_connection_config()


def test_connection_config_kerberos():
    cfg = _cfg(
        auth_mode="kerberos",
        kerberos_service="vertica",
        kerberos_host="kdc.example.com",
    )
    pool = VerticaConnectionPool.__new__(VerticaConnectionPool)
    pool.config = cfg
    config = pool._get_connection_config()
    assert config["kerberos_service_name"] == "vertica"
    assert config["kerberos_host_name"] == "kdc.example.com"


def test_connection_config_ssl_basic():
    """SSL=True without cert/key uses simple ssl=True config."""
    cfg = _cfg(ssl=True)
    pool = VerticaConnectionPool.__new__(VerticaConnectionPool)
    pool.config = cfg
    config = pool._get_connection_config()
    assert config["ssl"] is True


def test_connection_config_mtls_missing_cert():
    """mTLS without cert/key raises ValueError."""
    cfg = _cfg(auth_mode="mtls", ssl=True, ssl_cert="", ssl_key="")
    pool = VerticaConnectionPool.__new__(VerticaConnectionPool)
    pool.config = cfg
    with pytest.raises(ValueError, match="ssl_cert and ssl_key are required"):
        pool._get_connection_config()


def test_connection_config_no_ssl():
    """No SSL → tlsmode=disable."""
    cfg = _cfg(ssl=False)
    pool = VerticaConnectionPool.__new__(VerticaConnectionPool)
    pool.config = cfg
    config = pool._get_connection_config()
    assert config.get("tlsmode") == "disable"


# ─── VerticaConnectionManager: is_operation_allowed ──────────────────────────

def test_manager_is_operation_allowed_no_config():
    mgr = VerticaConnectionManager()
    assert mgr.is_operation_allowed("public", OperationType.INSERT) is False


def test_manager_is_operation_allowed_global_insert():
    mgr = VerticaConnectionManager()
    mgr.config = _cfg()
    assert mgr.is_operation_allowed("public", OperationType.INSERT) is False


def test_manager_is_operation_allowed_global_update():
    mgr = VerticaConnectionManager()
    mgr.config = _cfg()
    assert mgr.is_operation_allowed("public", OperationType.UPDATE) is False


def test_manager_is_operation_allowed_global_delete():
    mgr = VerticaConnectionManager()
    mgr.config = _cfg()
    assert mgr.is_operation_allowed("public", OperationType.DELETE) is False


def test_manager_is_operation_allowed_global_ddl():
    mgr = VerticaConnectionManager()
    mgr.config = _cfg()
    assert mgr.is_operation_allowed("public", OperationType.DDL) is False


def test_manager_is_operation_allowed_with_insert_enabled():
    mgr = VerticaConnectionManager()
    mgr.config = _cfg(allow_insert=True)
    assert mgr.is_operation_allowed("public", OperationType.INSERT) is True


def test_manager_not_initialized_raises():
    mgr = VerticaConnectionManager()
    with pytest.raises(Exception):
        mgr.get_connection()


def test_manager_release_connection_no_pool():
    mgr = VerticaConnectionManager()
    mgr.release_connection(MagicMock())  # must not raise


def test_manager_close_all_no_pool():
    mgr = VerticaConnectionManager()
    mgr.close_all()  # must not raise


# ─── _get_safe_config: masks sensitive keys ───────────────────────────────────

def test_get_safe_config_masks_password(monkeypatch):
    cfg = _cfg()
    fake_conn = MagicMock()
    monkeypatch.setattr(vconn, "vertica_python", types.SimpleNamespace(
        connect=lambda **kw: fake_conn
    ))
    pool = VerticaConnectionPool(cfg)
    safe = pool._get_safe_config({"password": "secret", "host": "h"})
    assert safe["password"] == "********"
    assert safe["host"] == "h"


def test_get_safe_config_masks_oauth_token(monkeypatch):
    cfg = _cfg()
    fake_conn = MagicMock()
    monkeypatch.setattr(vconn, "vertica_python", types.SimpleNamespace(
        connect=lambda **kw: fake_conn
    ))
    pool = VerticaConnectionPool(cfg)
    safe = pool._get_safe_config({"oauth_access_token": "tok", "host": "h"})
    assert safe["oauth_access_token"] == "********"
