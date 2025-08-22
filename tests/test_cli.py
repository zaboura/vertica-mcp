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

    def fake_run(*args, **kwargs):
        called["stdio"] = True

    # stdio path calls server.mcp.run
    monkeypatch.setattr(server.mcp, "run", fake_run)

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
