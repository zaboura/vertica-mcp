#!/usr/bin/env sh
set -e

# Default to stdio if not provided
TRANSPORT="${TRANSPORT:-stdio}"
PORT="${PORT:-8000}"
BIND="${BIND:-0.0.0.0}"
HTTP_PATH="${HTTP_PATH:-/mcp}"

case "$TRANSPORT" in
  stdio)
    # no ports, pure stdio
    exec vertica-mcp --transport stdio "$@"
    ;;
  sse)
    # SSE server
    exec vertica-mcp --transport sse --port "$PORT" --bind-host "$BIND" "$@"
    ;;
  http|streamable-http)
    # Streamable HTTP server
    exec vertica-mcp --transport http --port "$PORT" --bind-host "$BIND" --http-path "$HTTP_PATH" "$@"
    ;;
  *)
    echo "Unknown TRANSPORT: $TRANSPORT" >&2
    exit 1
    ;;
esac
