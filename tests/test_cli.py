# tests/test_cli.py
import pytest
from click.testing import CliRunner

# Import the command directly
from vertica_mcp.cli import cli as cli_cmd
import vertica_mcp.server as server


@pytest.fixture
def runner():
    return CliRunner()


def test_cli_chooses_stdio_and_calls_mcp_run(monkeypatch, runner):
    called = {}

    async def fake_run_stdio():
        called["stdio"] = True

    cb = cli_cmd.callback
    monkeypatch.setitem(cb.__globals__, "run_stdio", fake_run_stdio)

    res = runner.invoke(cli_cmd, ["--transport", "stdio"])
    assert res.exit_code == 0
    assert called.get("stdio") is True


def test_cli_chooses_sse(monkeypatch, runner):
    called = {}

    async def fake_run_sse(host="localhost", port=8000):
        called["sse"] = (host, port)

    # Patch the callback's globals so the function uses our fake
    cb = cli_cmd.callback  # click.Command -> underlying function
    monkeypatch.setitem(cb.__globals__, "run_sse", fake_run_sse)

    res = runner.invoke(cli_cmd, ["--transport", "sse", "--port", "7777"])
    assert res.exit_code == 0, res.output
    assert called["sse"] == ("localhost", 7777)


def test_cli_chooses_http(monkeypatch, runner):
    called = {}

    async def fake_run_http(host="localhost", port=8000, path="/mcp", json_response=False, stateless_http=True):
        called["http"] = (host, port, path, json_response, stateless_http)

    # Patch the callback's globals so the function uses our fake
    cb = cli_cmd.callback
    monkeypatch.setitem(cb.__globals__, "run_http", fake_run_http)

    res = runner.invoke(
        cli_cmd,
        ["--transport", "http", "--port", "9999", "--http-path", "/mcp", "--http-json", "--http-stateless"],
    )
    assert res.exit_code == 0, res.output
    assert called["http"] == ("localhost", 9999, "/mcp", True, True)

def test_cli_init_creates_env_file(runner):
    with runner.isolated_filesystem():
        res = runner.invoke(cli_cmd, ["--init"])
        assert res.exit_code == 0
        import os
        assert os.path.exists(".env")
        with open(".env", "r") as f:
            content = f.read()
            assert "VERTICA_HOST=localhost" in content

def test_cli_init_already_exists_yes(runner):
    with runner.isolated_filesystem():
        with open(".env", "w") as f:
            f.write("OLD_CONTENT")
        
        # User says yes ('y') to overwrite
        res = runner.invoke(cli_cmd, ["--init"], input="y\n")
        assert res.exit_code == 0
        with open(".env", "r") as f:
            content = f.read()
            assert "VERTICA_HOST=localhost" in content

def test_cli_init_already_exists_no(runner):
    with runner.isolated_filesystem():
        with open(".env", "w") as f:
            f.write("OLD_CONTENT")
        
        # User says no ('n') to overwrite
        res = runner.invoke(cli_cmd, ["--init"], input="n\n")
        assert res.exit_code == 0
        with open(".env", "r") as f:
            content = f.read()
            assert content == "OLD_CONTENT"

def test_cli_env_overrides(monkeypatch, runner):
    called = {}
    
    async def fake_run_stdio():
        called["stdio"] = True

    cb = cli_cmd.callback
    monkeypatch.setitem(cb.__globals__, "run_stdio", fake_run_stdio)
    
    import os
    res = runner.invoke(cli_cmd, [
        "--transport", "stdio",
        "--host", "my-vertica-host",
        "--db-port", "1234",
        "--database", "mydb",
        "--user", "myuser",
        "--password", "mypass",
        "--connection-limit", "25",
        "--ssl",
        "--ssl-reject-unauthorized"
    ])
    
    assert res.exit_code == 0
    assert called.get("stdio") is True
    assert os.environ.get("VERTICA_HOST") == "my-vertica-host"
    assert os.environ.get("VERTICA_PORT") == "1234"
    assert os.environ.get("VERTICA_DATABASE") == "mydb"
    assert os.environ.get("VERTICA_USER") == "myuser"
    assert os.environ.get("VERTICA_PASSWORD") == "mypass"
    assert os.environ.get("VERTICA_CONNECTION_LIMIT") == "25"
    assert os.environ.get("VERTICA_SSL") == "true"

