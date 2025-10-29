# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.11-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Copy metadata + source (dynamic version needs the package present)
COPY pyproject.toml uv.lock* README.md ./
COPY vertica_mcp ./vertica_mcp

# Install runtime deps
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

# Make venv default
ENV PATH="/app/.venv/bin:${PATH}"

# Install project (for CLI entrypoint)
RUN uv pip install --no-cache-dir -e .

# Entrypoint (normalize EOLs for Windows-originated files)
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh \
 && chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
