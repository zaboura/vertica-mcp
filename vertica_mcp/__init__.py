__version__ = "0.1.0"

# from vertica_mcp.server import mcp, run_sse
from vertica_mcp.cli import main, cli

__all__ = ["main", "cli"]
