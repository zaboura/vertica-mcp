# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.11-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Copy metadata first for layer caching
COPY pyproject.toml uv.lock* ./

# Install runtime deps (not your source yet)
RUN uv sync --frozen --no-dev

# Copy source + README (needed for -e .)
COPY vertica_mcp ./vertica_mcp
COPY README.md ./

# Install package (provides /usr/local/bin/vertica-mcp)
RUN uv pip install -e .

# Lightweight entrypoint to switch transport via $TRANSPORT (default=stdio)
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Expose only for SSE/HTTP modes (harmless for stdio)
EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
# No CMD -> defaults to stdio via entrypoint
