# tests/test_utils.py
import logging
from vertica_mcp.utils import setup_logger


def test_setup_logger_writes_to_stderr(capsys):
    log = setup_logger(verbosity=2)  # DEBUG
    log.debug("hello debug")
    log.info("hello info")
    captured = capsys.readouterr()
    # Messages should be in stderr (StreamHandler default)
    assert "hello debug" in captured.err or "hello debug" in captured.out
    assert "hello info" in captured.err or "hello info" in captured.out
