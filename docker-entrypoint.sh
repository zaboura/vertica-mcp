#!/usr/bin/env sh
set -e

# Defaults (safe across OS/CI)
TRANSPORT="${TRANSPORT:-stdio}"
PORT="${PORT:-8000}"
BIND="${BIND:-0.0.0.0}"
HTTP_PATH="${HTTP_PATH:-/mcp}"

# Allow bypassing DB check explicitly (e.g., demos)
SKIP_DB_CHECK="${SKIP_DB_CHECK:-0}"

# ---- Pre-flight: Vertica credentials must be present ----
# Required for all transports, since the server connects to Vertica.
# (You can override by setting SKIP_DB_CHECK=1)
if [ "$SKIP_DB_CHECK" != "1" ]; then
  REQUIRED_VARS="VERTICA_HOST VERTICA_USER VERTICA_PASSWORD"
  MISSING=0
  for VAR in $REQUIRED_VARS; do
    # shellcheck disable=SC2086
    if [ -z "$(eval echo \$$VAR)" ]; then
      echo "❌ ERROR: Missing required environment variable: $VAR" >&2
      MISSING=1
    fi
  done
  if [ "$MISSING" -eq 1 ]; then
    echo >&2
    echo "Provide Vertica credentials via .env, Compose env, or -e flags, for example:" >&2
    echo "  docker run -it --rm \\" >&2
    echo "    -e TRANSPORT=http -p 8000:8000 \\" >&2
    echo "    -e VERTICA_HOST=my.vertica.server \\" >&2
    echo "    -e VERTICA_USER=dbuser \\" >&2
    echo "    -e VERTICA_PASSWORD=supersecret \\" >&2
    echo "    vertica-mcp:latest" >&2
    echo >&2
    echo "Or create a .env file and run: docker compose up mcp-http" >&2
    exit 1
  fi
fi

echo "🔧 MCP transport: $TRANSPORT"
case "$TRANSPORT" in
  stdio)
    echo "   mode: stdio"
    exec vertica-mcp --transport stdio "$@"
    ;;
  sse)
    echo "   mode: sse  bind=$BIND  port=$PORT"
    exec vertica-mcp --transport sse --port "$PORT" --bind-host "$BIND" "$@"
    ;;
  http|streamable-http)
    echo "   mode: http bind=$BIND  port=$PORT  path=$HTTP_PATH"
    exec vertica-mcp --transport http --port "$PORT" --bind-host "$BIND" --http-path "$HTTP_PATH" "$@"
    ;;
  *)
    echo "❌ Unknown TRANSPORT: $TRANSPORT" >&2
    echo "   Use one of: stdio | http | sse" >&2
    exit 1
    ;;
esac
