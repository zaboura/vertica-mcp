import pytest
from unittest.mock import patch, MagicMock
from starlette.testclient import TestClient
from vertica_mcp.server import mcp

@pytest.fixture
def client():
    # Setup test client with HTTP transport
    app = mcp.streamable_http_app()
    from vertica_mcp.server import AuthMiddleware
    app.add_middleware(AuthMiddleware)
    return TestClient(app)

@patch("os.getenv")
def test_auth_middleware_missing_token(mock_getenv, client):
    mock_getenv.side_effect = lambda k, d=None: "https://issuer/" if k == "JWT_ISSUER" else "aud" if k == "JWT_AUDIENCE" else d
    
    response = client.post("/mcp/messages", json={})
    assert response.status_code == 401
    assert "error" in response.json()
