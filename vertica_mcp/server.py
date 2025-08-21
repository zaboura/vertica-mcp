"""Server module for Vertica MCP. Provides API endpoints and database utilities."""

import asyncio
import logging
import os
import re
import socket
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional

import uvicorn
from dotenv import find_dotenv, load_dotenv
from mcp.server.fastmcp import Context, FastMCP

from vertica_mcp.connection import (OperationType, VerticaConfig,
                                    VerticaConnectionManager)

MCP_SERVER_NAME = "vertica-mcp"
DEPENDENCIES = ["vertica-python", "pydantic", "starlette", "uvicorn"]

# Configure logging
logger = logging.getLogger("vertica-mcp")

# Configuration from environment
QUERY_TIMEOUT = int(os.getenv("VERTICA_QUERY_TIMEOUT", "600"))  # 10 minutes default
MAX_RETRY_ATTEMPTS = int(os.getenv("VERTICA_MAX_RETRIES", "3"))
RETRY_DELAY_BASE = float(os.getenv("VERTICA_RETRY_DELAY", "1.0"))
CACHE_TTL_SECONDS = int(os.getenv("VERTICA_CACHE_TTL", "300"))  # 5 minutes
MAX_RESULT_SIZE_MB = int(os.getenv("VERTICA_MAX_RESULT_MB", "100"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("VERTICA_RATE_LIMIT", "60"))
CONNECTION_HEALTH_CHECK_INTERVAL = int(os.getenv("VERTICA_HEALTH_CHECK_INTERVAL", "60"))

# Cache for metadata queries
metadata_cache: Dict[str, tuple[Any, float]] = {}
# Rate limiting tracking
rate_limit_tracker: Dict[str, List[float]] = {}


def _strip_sql_comments(q: str) -> str:
    # remove /* ... */ and -- ... EOL
    q = re.sub(r"/\*.*?\*/", "", q, flags=re.S)
    q = re.sub(r"--[^\n]*", "", q)
    return q


def _is_select(query: str) -> bool:
    """SELECT-like statements (WITH/SELECT/EXPLAIN/PROFILE)."""
    q = _strip_sql_comments(query).strip()
    while q.startswith("(") and q.endswith(")"):
        q = q[1:-1].strip()
    return re.match(r"^(WITH|SELECT|EXPLAIN|PROFILE)\b", q, flags=re.I) is not None


def _wrap_subquery(sql: str) -> str:
    """Wrap a subquery in a SELECT statement."""
    sql = sql.replace(";", "").strip()
    return f"SELECT * FROM ({sql}) q"


def _sanitize_query(query: str) -> str:
    """Basic SQL injection prevention."""
    # Check for common injection patterns
    dangerous_patterns = [
        r";\s*DROP\s+",
        r";\s*DELETE\s+",
        r";\s*UPDATE\s+",
        r";\s*INSERT\s+",
        r";\s*ALTER\s+",
        r";\s*CREATE\s+",
        r"xp_cmdshell",
        r"sp_executesql",
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            raise ValueError(f"Potentially dangerous SQL pattern detected: {pattern}")

    return query


def _check_rate_limit(client_id: str) -> bool:
    """Check if client has exceeded rate limit."""
    now = time.time()
    if client_id not in rate_limit_tracker:
        rate_limit_tracker[client_id] = []

    # Clean old entries
    rate_limit_tracker[client_id] = [
        t for t in rate_limit_tracker[client_id] if now - t < 60
    ]

    if len(rate_limit_tracker[client_id]) >= RATE_LIMIT_PER_MINUTE:
        return False

    rate_limit_tracker[client_id].append(now)
    return True


@lru_cache(maxsize=128)
def _get_cached_metadata(cache_key: str) -> Optional[Any]:
    """Get cached metadata if not expired."""
    if cache_key in metadata_cache:
        data, timestamp = metadata_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL_SECONDS:
            return data
        else:
            del metadata_cache[cache_key]
    return None


def _set_cached_metadata(cache_key: str, data: Any):
    """Set metadata cache with timestamp."""
    metadata_cache[cache_key] = (data, time.time())


async def _validate_connection(conn) -> bool:
    """Validate database connection is alive."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        return True
    except Exception:
        return False


def extract_operation_type(query: str) -> OperationType | None:
    """Extract the operation type from a SQL query."""
    query = query.strip().upper()

    if query.startswith("INSERT"):
        return OperationType.INSERT
    elif query.startswith("UPDATE"):
        return OperationType.UPDATE
    elif query.startswith("DELETE"):
        return OperationType.DELETE
    elif any(query.startswith(op) for op in ["CREATE", "ALTER", "DROP", "TRUNCATE"]):
        return OperationType.DDL
    return None


def extract_schema_from_query(query: str) -> str | None:
    """Extract the schema from a SQL query."""
    q = query.strip().lower()
    m = re.search(r"([a-zA-Z0-9_]+)\.[a-zA-Z0-9_]+", q)
    if m:
        return m.group(1)
    return None


async def run_sse(host: str = "localhost", port: int = 8000) -> None:
    """Launch the MCP server with HTTP-SSE transport."""
    logger.info(f"Starting MCP server with SSE transport on {host}:{port}")

    sse_app = mcp.sse_app()

    print(f"\n╔{'═' * 50}╗")
    print(f"║{'Vertica MCP Server':^50}║")
    print(f"╠{'═' * 50}╣")
    print(f"║  Transport : SSE{' ' * 33}║")
    print(
        f"║  Endpoint  : http://{host}:{port}{' ' * (28 - len(host) - len(str(port)))}║"
    )
    print(f"║  Status    : Ready{' ' * 31}║")
    print(f"╚{'═' * 50}╝\n")
    print(f"📝 Connect MCP clients to: http://{host}:{port}/sse")

    config = uvicorn.Config(
        sse_app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        use_colors=True,
        timeout_keep_alive=30,  # Add keepalive timeout
        limit_max_requests=1000,  # Restart workers after N requests
    )

    server = uvicorn.Server(config)

    try:
        await server.serve()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down gracefully")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


async def run_http(
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/mcp",
    json_response: bool = False,
    stateless_http: bool = True,
) -> None:
    """Launch the MCP server with Streamable HTTP transport."""
    logger.info(f"Starting MCP server with Streamable HTTP on {host}:{port}{path}")

    mcp.settings.host = host
    mcp.settings.port = port
    mcp.settings.streamable_http_path = path
    mcp.settings.json_response = json_response
    mcp.settings.stateless_http = stateless_http

    http_app = mcp.streamable_http_app()

    print(f"\n╔{'═' * 50}╗")
    print(f"║{'Vertica MCP Server':^50}║")
    print(f"╠{'═' * 50}╣")
    print(f"║  Transport : Streamable HTTP{' ' * 21}║")
    ep = f"http://{host}:{port}{path}"
    pad = max(0, 36 - len(ep))
    print(f"║  Endpoint  : {ep}{' ' * pad}║")
    print(f"║  Status    : Ready{' ' * 31}║")
    print(f"╚{'═' * 50}╝\n")

    config = uvicorn.Config(
        http_app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        use_colors=True,
        timeout_keep_alive=30,
        limit_max_requests=1000,
    )

    server = uvicorn.Server(config)
    try:
        await server.serve()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down gracefully")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


@asynccontextmanager
async def server_lifespan(_server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """Server lifespan context manager with improved error handling."""

    # Load environment with multiple fallback methods
    env_loaded = False
    env_methods = [
        lambda: find_dotenv(usecwd=True),
        lambda: os.path.join(os.getcwd(), ".env"),
        lambda: os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
        ),
    ]

    for method in env_methods:
        try:
            env_path = method()
            if env_path and os.path.exists(env_path):
                load_dotenv(env_path, override=False)
                logger.info(f"Loaded environment from {env_path}")
                env_loaded = True
                break
        except Exception as e:
            logger.debug(f"Environment loading attempt failed: {e}")

    if not env_loaded:
        load_dotenv()  # Try default
        logger.warning("Using system environment variables or defaults")

    manager = None
    retry_count = 0
    max_init_retries = 3

    while retry_count < max_init_retries:
        try:
            manager = VerticaConnectionManager()
            config = VerticaConfig.from_env()

            # Validate configuration
            if not config.host or not config.database:
                raise ValueError("Missing required database configuration")

            logger.info(
                f"Vertica config: host={config.host} port={config.port} "
                f"db={config.database} user={config.user} ssl={config.ssl}"
            )

            # Test TCP connectivity
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                result = sock.connect_ex((config.host, config.port))
                sock.close()

                if result != 0:
                    raise ConnectionError(f"TCP connection failed with code {result}")

                logger.info("✅ TCP connectivity successful")
            except Exception as e:
                logger.error(f"❌ Network connectivity test failed: {e}")
                raise

            # Initialize connection manager
            manager.initialize_default(config)
            logger.info("Connection manager initialized")

            # Test database connection
            conn = None
            try:
                conn = manager.get_connection()
                if await _validate_connection(conn):
                    cursor = conn.cursor()
                    cursor.execute("SELECT version()")
                    version = cursor.fetchone()[0]
                    logger.info(f"✅ Connected to Vertica: {version}")
                    cursor.close()
                    break  # Success
                else:
                    raise ConnectionError("Connection validation failed")

            finally:
                if conn:
                    manager.release_connection(conn)

        except Exception as e:
            retry_count += 1
            logger.error(
                f"Initialization attempt {retry_count}/{max_init_retries} failed: {e}"
            )

            if retry_count < max_init_retries:
                await asyncio.sleep(2**retry_count)  # Exponential backoff
            else:
                logger.error("Failed to initialize after all retries")
                # Continue anyway but mark as degraded

    # Start background health check task
    health_check_task = None
    if manager:

        async def health_check_loop():
            while True:
                try:
                    await asyncio.sleep(CONNECTION_HEALTH_CHECK_INTERVAL)
                    conn = manager.get_connection()
                    if not await _validate_connection(conn):
                        logger.warning("Health check failed, attempting reconnection")
                        # Trigger reconnection logic here
                    manager.release_connection(conn)
                except Exception as e:
                    logger.error(f"Health check error: {e}")

        health_check_task = asyncio.create_task(health_check_loop())

    try:
        yield {"vertica_manager": manager, "health_task": health_check_task}
    finally:
        if health_check_task:
            health_check_task.cancel()
        if manager:
            try:
                manager.close_all()
                logger.info("Closed all Vertica connections")
            except Exception as e:
                logger.error(f"Error closing connections: {e}")


# Create FastMCP instance with SSE support
mcp = FastMCP(
    MCP_SERVER_NAME,
    dependencies=DEPENDENCIES,
    lifespan=server_lifespan,
    stateless_http=True,
)


#--------------------------------------------
#---------- Run Query Safely ------------------
#--------------------------------------------
@mcp.tool()
async def run_query_safely(
    ctx: Context,
    query: str,
    row_threshold: int = 1000,
    proceed: bool = False,
    mode: str = "page",
    page_limit: int = 2000,
    include_columns: bool = True,
    precount: bool = False,
    timeout: Optional[int] = None,
) -> dict:
    """
    Safe query execution with size detection, pagination, and timeout support.

    Args:
        query: SQL query to execute
        row_threshold: Maximum rows before requiring confirmation
        proceed: Whether to proceed with large result set
        mode: Execution mode ('page' or 'stream')
        page_limit: Rows per page when paginating
        include_columns: Include column names in response
        precount: Count total rows for large results (expensive)
        timeout: Query timeout in seconds (default from env)
    """
    await ctx.info("run_query_safely called")

    # Rate limiting check
    client_id = (
        getattr(ctx.request_context, "client_id", None)
        or getattr(ctx.request_context, "connection_id", None)
        or "default"
    )

    if not _check_rate_limit(client_id):
        raise RuntimeError("Rate limit exceeded. Please wait before retrying.")

    # Sanitize query
    try:
        query = _sanitize_query(query)
    except ValueError as e:
        await ctx.error(f"Query validation failed: {e}")
        raise RuntimeError(f"Query validation failed: {e}")

    manager = ctx.request_context.lifespan_context.get("vertica_manager")
    if not manager:
        raise RuntimeError("No database connection manager available")

    # Permission checks
    schema = extract_schema_from_query(query)
    operation = extract_operation_type(query)
    if operation and not manager.is_operation_allowed(schema or "default", operation):
        msg = f"Operation {operation.name} not allowed for schema {schema}"
        await ctx.error(msg)
        raise RuntimeError(msg)

    # Set timeout
    query_timeout = timeout or QUERY_TIMEOUT

    # Non-SELECT execution
    if not _is_select(query):
        conn = cursor = None
        try:
            conn = manager.get_connection()

            # Validate connection before use
            if not await _validate_connection(conn):
                manager.release_connection(conn)
                conn = manager.get_connection()  # Get fresh connection

            cursor = conn.cursor()

            # Set query timeout
            cursor.execute(f"SET SESSION RUNTIMECAP '{query_timeout}s'")

            # Execute with timeout
            cursor.execute(query)
            affected = getattr(cursor, "rowcount", None)

            # Commit for DML operations
            if operation in [
                OperationType.INSERT,
                OperationType.UPDATE,
                OperationType.DELETE,
            ]:
                conn.commit()

            await ctx.info(f"Non-SELECT executed, affected_rows={affected}")
            return {"ok": True, "affected_rows": affected}

        except Exception as e:
            if conn and operation:
                conn.rollback()  # Rollback on error
            msg = f"Error executing statement: {e}"
            await ctx.error(msg)
            raise RuntimeError(msg) from e
        finally:
            if cursor:
                cursor.close()
            if conn:
                manager.release_connection(conn)

    # SELECT query handling
    if not proceed:
        # Probe for size
        probe_limit = row_threshold + 1
        probe_sql = f"{_wrap_subquery(query)} LIMIT {probe_limit}"

        conn = cursor = None
        try:
            conn = manager.get_connection()

            if not await _validate_connection(conn):
                manager.release_connection(conn)
                conn = manager.get_connection()

            cursor = conn.cursor()
            cursor.execute(f"SET SESSION RUNTIMECAP '{query_timeout}s'")
            cursor.execute(probe_sql)

            rows = cursor.fetchall()
            cols = (
                [d[0] for d in cursor.description]
                if include_columns and cursor.description
                else None
            )

            is_large = len(rows) > row_threshold
            preview = rows[: min(50, len(rows))]

            exact_count = None
            if is_large and precount:
                await ctx.info("Computing exact COUNT(*)")
                cursor.execute(f"SELECT COUNT(*) FROM ({query}) q")
                exact_count = int(cursor.fetchone()[0])

            if not is_large:
                await ctx.info(f"Small result (<= {row_threshold})")
                return {
                    "ok": True,
                    "rows": rows,
                    "count": len(rows),
                    "done": True,
                    "columns": cols,
                    "large": False,
                }

            # Large result - require confirmation
            human_msg = (
                f"Large result detected (> {row_threshold} rows)"
                + (f": about {exact_count} rows." if exact_count else ".")
                + " Proceed?"
            )
            await ctx.warning(human_msg)

            return {
                "ok": True,
                "large": True,
                "requires_confirmation": True,
                "threshold": row_threshold,
                "exact_count": exact_count,
                "message": human_msg,
                "preview": preview,
                "columns": cols,
                "next_step": {
                    "tool": "run_query_safely",
                    "arguments": {
                        "query": query,
                        "row_threshold": row_threshold,
                        "proceed": True,
                        "mode": "page",
                        "page_limit": page_limit,
                        "include_columns": include_columns,
                    },
                },
            }

        except Exception as e:
            msg = f"Error probing query: {e}"
            await ctx.error(msg)
            raise RuntimeError(msg) from e
        finally:
            if cursor:
                cursor.close()
            if conn:
                manager.release_connection(conn)

    # Proceed with large result
    await ctx.info(f"Proceeding with mode={mode}")

    if mode == "page":
        return await execute_query_paginated(
            ctx=ctx,
            query=query,
            limit=page_limit,
            offset=0,
            include_columns=include_columns,
            timeout=query_timeout,
        )
    elif mode == "stream":
        return await execute_query_stream(
            ctx=ctx,
            query=query,
            batch_size=max(page_limit, 1000),
            timeout=query_timeout,
        )
    else:
        raise RuntimeError(f"Unknown mode: {mode}")


#--------------------------------------------
#---------- Execute Query Paginated ------------------
#--------------------------------------------
@mcp.tool()
async def execute_query_paginated(
    ctx: Context,
    query: str,
    limit: int = 2000,
    offset: int = 0,
    include_columns: bool = True,
    timeout: Optional[int] = None,
) -> dict:
    """Execute query with pagination support and result size limits."""
    await ctx.info(f"Paginated query: limit={limit}, offset={offset}")

    manager = ctx.request_context.lifespan_context.get("vertica_manager")
    if not manager:
        raise RuntimeError("No database connection manager available")

    # Validate it's a SELECT
    op = extract_operation_type(query)
    if op:
        raise RuntimeError("Paginated execution only supports SELECT statements")

    paged_sql = f"{_wrap_subquery(query)} LIMIT {int(limit)} OFFSET {int(offset)}"
    query_timeout = timeout or QUERY_TIMEOUT

    conn = cursor = None
    try:
        conn = manager.get_connection()

        if not await _validate_connection(conn):
            manager.release_connection(conn)
            conn = manager.get_connection()

        cursor = conn.cursor()
        cursor.execute(f"SET SESSION RUNTIMECAP '{query_timeout}s'")
        cursor.execute(paged_sql)

        rows = cursor.fetchall()
        cols = (
            [d[0] for d in cursor.description]
            if include_columns and cursor.description
            else None
        )

        # Check result size
        import sys

        result_size = sys.getsizeof(rows) / (1024 * 1024)  # MB
        if result_size > MAX_RESULT_SIZE_MB:
            await ctx.warning(f"Result size ({result_size:.2f}MB) exceeds limit")
            rows = rows[: len(rows) // 2]  # Truncate to half

        done = len(rows) < limit

        return {
            "rows": rows,
            "count": len(rows),
            "next_offset": offset + len(rows),
            "done": done,
            "columns": cols,
        }

    except Exception as e:
        msg = f"Paginated query error: {str(e)}"
        await ctx.error(msg)
        raise RuntimeError(msg) from e
    finally:
        if cursor:
            cursor.close()
        if conn:
            manager.release_connection(conn)

#--------------------------------------------
#---------- Execute Query Stream ------------------
#--------------------------------------------
@mcp.tool()
async def execute_query_stream(
    ctx: Context,
    query: str,
    batch_size: int = 1000,
    max_rows: int = 100000,
    timeout: Optional[int] = None,
) -> dict:
    """Stream query results with batching and size limits."""
    await ctx.info(f"Streaming query with batch_size={batch_size}")

    manager = ctx.request_context.lifespan_context.get("vertica_manager")
    if not manager:
        raise RuntimeError("No database connection manager available")

    schema = extract_schema_from_query(query)
    operation = extract_operation_type(query)
    if operation and not manager.is_operation_allowed(schema or "default", operation):
        raise RuntimeError(f"Operation {operation.name} not allowed")

    query_timeout = timeout or QUERY_TIMEOUT
    conn = cursor = None

    try:
        conn = manager.get_connection()

        if not await _validate_connection(conn):
            manager.release_connection(conn)
            conn = manager.get_connection()

        cursor = conn.cursor()
        cursor.execute(f"SET SESSION RUNTIMECAP '{query_timeout}s'")
        cursor.execute(query)

        all_results = []
        total_rows = 0
        total_size_mb = 0

        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break

            # Check size limits
            import sys

            batch_size_mb = sys.getsizeof(batch) / (1024 * 1024)
            total_size_mb += batch_size_mb

            if total_size_mb > MAX_RESULT_SIZE_MB:
                await ctx.warning(f"Result size limit reached ({MAX_RESULT_SIZE_MB}MB)")
                break

            total_rows += len(batch)
            all_results.extend(batch)

            await ctx.debug(f"Fetched {total_rows} rows ({total_size_mb:.2f}MB)")

            if total_rows >= max_rows:
                await ctx.warning(f"Row limit reached ({max_rows})")
                break

        await ctx.info(f"Stream complete: {total_rows} rows, {total_size_mb:.2f}MB")

        return {
            "result": all_results,
            "total_rows": total_rows,
            "truncated": total_rows >= max_rows or total_size_mb >= MAX_RESULT_SIZE_MB,
            "size_mb": round(total_size_mb, 2),
        }

    except Exception as e:
        error_msg = f"Stream error: {str(e)}"
        await ctx.error(error_msg)
        raise RuntimeError(error_msg) from e
    finally:
        if cursor:
            cursor.close()
        if conn:
            manager.release_connection(conn)


#--------------------------------------------
#---------- Get Table Structure ------------------
#--------------------------------------------
@mcp.tool()
async def get_table_structure(
    ctx: Context, table_name: str, schema_name: str = "public"
) -> dict:
    """Get table structure with caching support."""
    cache_key = f"table_structure:{schema_name}.{table_name}"
    cached = _get_cached_metadata(cache_key)
    if cached:
        await ctx.info(f"Using cached structure for {schema_name}.{table_name}")
        return cached

    await ctx.info(f"Fetching structure for {schema_name}.{table_name}")

    manager = ctx.request_context.lifespan_context.get("vertica_manager")
    if not manager:
        raise RuntimeError("No database connection manager available")

    query = """
    SELECT column_name, data_type, character_maximum_length,
           numeric_precision, numeric_scale, is_nullable, column_default
    FROM v_catalog.columns
    WHERE table_schema = %s AND table_name = %s
    ORDER BY ordinal_position;
    """

    conn = cursor = None
    try:
        conn = manager.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, (schema_name, table_name))
        columns = cursor.fetchall()

        if not columns:
            raise RuntimeError(f"Table not found: {schema_name}.{table_name}")

        # Get constraints
        cursor.execute(
            """
            SELECT constraint_name, constraint_type, column_name
            FROM v_catalog.constraint_columns
            WHERE table_schema = %s AND table_name = %s;
        """,
            (schema_name, table_name),
        )
        constraints = cursor.fetchall()

        # Format result
        result = f"Table: {schema_name}.{table_name}\n\nColumns:\n"
        for col in columns:
            result += f"- {col[0]}: {col[1]}"
            if col[2]:
                result += f"({col[2]})"
            elif col[3]:
                result += f"({col[3]},{col[4]})"
            result += f" {'NULL' if col[5] == 'YES' else 'NOT NULL'}"
            if col[6]:
                result += f" DEFAULT {col[6]}"
            result += "\n"

        if constraints:
            result += "\nConstraints:\n"
            for const in constraints:
                result += f"- {const[0]} ({const[1]}): {const[2]}\n"

        response = {
            "result": result,
            "table_name": table_name,
            "schema": schema_name,
            "column_count": len(columns),
            "constraint_count": len(constraints),
        }

        _set_cached_metadata(cache_key, response)
        return response

    except Exception as e:
        error_msg = f"Error getting table structure: {str(e)}"
        await ctx.error(error_msg)
        raise RuntimeError(error_msg) from e
    finally:
        if cursor:
            cursor.close()
        if conn:
            manager.release_connection(conn)


@mcp.tool()
async def get_table_projections(
    ctx: Context, table_name: str, schema_name: str = "public"
) -> dict:
    """List projections for a table."""
    await ctx.info(f"Listing projections for {schema_name}.{table_name}")

    manager = ctx.request_context.lifespan_context.get("vertica_manager")
    if not manager:
        raise RuntimeError("No database connection manager available")

    query = """
    SELECT projection_name, is_super_projection, anchor_table_name, create_type
    FROM v_catalog.projections
    WHERE projection_schema = %s AND anchor_table_name = %s
    ORDER BY projection_name;
    """

    conn = cursor = None
    try:
        conn = manager.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, (schema_name, table_name))
        projections = cursor.fetchall()

        if not projections:
            raise RuntimeError(f"No projections found for {schema_name}.{table_name}")

        result = f"Projections for {schema_name}.{table_name}:\n\n"
        for proj in projections:
            result += f"- {proj[0]} (Super: {proj[1]}, Type: {proj[3]})\n"

        return {
            "result": result,
            "table_name": table_name,
            "schema": schema_name,
            "projection_count": len(projections),
        }

    except Exception as e:
        error_msg = f"Error listing projections: {str(e)}"
        await ctx.error(error_msg)
        raise RuntimeError(error_msg) from e
    finally:
        if cursor:
            cursor.close()
        if conn:
            manager.release_connection(conn)


#--------------------------------------------
#---------- Get Schema Views ------------------
#--------------------------------------------
@mcp.tool()
async def get_schema_views(ctx: Context, schema_name: str = "public") -> dict:
    """List views in schema with caching."""
    cache_key = f"schema_views:{schema_name}"
    cached = _get_cached_metadata(cache_key)
    if cached:
        return cached

    await ctx.info(f"Listing views in schema: {schema_name}")

    manager = ctx.request_context.lifespan_context.get("vertica_manager")
    if not manager:
        raise RuntimeError("No database connection manager available")

    query = """
    SELECT table_name, view_definition
    FROM v_catalog.views
    WHERE table_schema = %s
    ORDER BY table_name;
    """

    conn = cursor = None
    try:
        conn = manager.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, (schema_name,))
        views = cursor.fetchall()

        if not views:
            raise RuntimeError(f"No views found in schema: {schema_name}")

        result = f"Views in {schema_name}:\n\n"
        for view in views:
            result += f"- {view[0]}\n"

        response = {"result": result, "schema": schema_name, "view_count": len(views)}
        _set_cached_metadata(cache_key, response)
        return response

    except Exception as e:
        error_msg = f"Error listing views: {str(e)}"
        await ctx.error(error_msg)
        raise RuntimeError(error_msg) from e
    finally:
        if cursor:
            cursor.close()
        if conn:
            manager.release_connection(conn)


#--------------------------------------------
#---------- Get Schema Tables ------------------
#--------------------------------------------
@mcp.tool()
async def get_schema_tables(ctx: Context, schema_name: str = "public") -> dict:
    """List tables in schema with caching."""
    cache_key = f"schema_tables:{schema_name}"
    cached = _get_cached_metadata(cache_key)
    if cached:
        return cached

    await ctx.info(f"Listing tables in schema: {schema_name}")

    manager = ctx.request_context.lifespan_context.get("vertica_manager")
    if not manager:
        raise RuntimeError("No database connection manager available")

    query = """
    SELECT table_name
    FROM v_catalog.tables
    WHERE table_schema = %s
    ORDER BY table_name;
    """

    conn = cursor = None
    try:
        conn = manager.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, (schema_name,))
        tables = cursor.fetchall()

        if not tables:
            raise RuntimeError(f"No tables found in schema: {schema_name}")

        result = f"Tables in {schema_name}:\n\n"
        for table in tables:
            result += f"- {table[0]}\n"

        response = {"result": result, "schema": schema_name, "table_count": len(tables)}
        _set_cached_metadata(cache_key, response)
        return response

    except Exception as e:
        error_msg = f"Error listing tables: {str(e)}"
        await ctx.error(error_msg)
        raise RuntimeError(error_msg) from e
    finally:
        if cursor:
            cursor.close()
        if conn:
            manager.release_connection(conn)

#--------------------------------------------
#---------- Get Database Schemas ------------------
#--------------------------------------------
@mcp.tool()
async def get_database_schemas(ctx: Context) -> dict:
    """List database schemas with caching."""
    cache_key = "database_schemas"
    cached = _get_cached_metadata(cache_key)
    if cached:
        return cached

    await ctx.info("Listing database schemas")

    manager = ctx.request_context.lifespan_context.get("vertica_manager")
    if not manager:
        raise RuntimeError("No database connection manager available")

    query = """
    SELECT schema_name, is_system_schema
    FROM v_catalog.schemata
    ORDER BY schema_name;
    """

    conn = cursor = None
    try:
        conn = manager.get_connection()
        cursor = conn.cursor()
        cursor.execute(query)
        schemas = cursor.fetchall()

        if not schemas:
            raise RuntimeError("No schemas found")

        result = "Database schemas:\n\n"
        for schema in schemas:
            result += f"- {schema[0]} {'(system)' if schema[1] else ''}\n"

        response = {"result": result, "schema_count": len(schemas)}
        _set_cached_metadata(cache_key, response)
        return response

    except Exception as e:
        error_msg = f"Error listing schemas: {str(e)}"
        await ctx.error(error_msg)
        raise RuntimeError(error_msg) from e
    finally:
        if cursor:
            cursor.close()
        if conn:
            manager.release_connection(conn)

#--------------------------------------------
#---------- Profile Query ------------------
#--------------------------------------------
def _inject_label(sql: str, label: str) -> str:
    """Insert /*+LABEL('...')*/ after the first top-level SELECT."""
    # 1) Strip any existing label hints
    sql = re.sub(
        r"/\*\+\s*label\s*\(\s*(['\"]).*?\1\s*\)\s*\*/",
        "",
        sql,
        flags=re.I | re.S,
    )

    s = sql
    n = len(s)
    i = 0
    depth = 0
    in_sq = False  # '...'
    in_dq = False  # "..."
    in_block = False  # /* ... */
    in_line = False   # -- ...

    def is_word_char(ch: str) -> bool:
        return ch.isalnum() or ch == "_"

    while i < n:
        ch = s[i]

        # line comment
        if not (in_sq or in_dq or in_block) and ch == "-" and i + 1 < n and s[i+1] == "-":
            in_line = True
            i += 2
            while i < n and s[i] not in "\r\n":
                i += 1
            in_line = False
            i += 1
            continue

        # block comment
        if not (in_sq or in_dq or in_line) and ch == "/" and i + 1 < n and s[i+1] == "*":
            in_block = True
            i += 2
            while i < n - 1:
                if s[i] == "*" and s[i+1] == "/":
                    in_block = False
                    i += 2
                    break
                i += 1
            continue

        if in_line or in_block:
            continue

        # strings
        if not in_dq and ch == "'":
            # handle escaped ''
            if in_sq and i + 1 < n and s[i+1] == "'":
                i += 2
                continue
            in_sq = not in_sq
            i += 1
            continue

        if not in_sq and ch == '"':
            # handle escaped ""
            if in_dq and i + 1 < n and s[i+1] == '"':
                i += 2
                continue
            in_dq = not in_dq
            i += 1
            continue

        if in_sq or in_dq:
            i += 1
            continue

        # parens
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            if depth > 0:
                depth -= 1
            i += 1
            continue

        # first top-level SELECT
        if depth == 0 and (ch == "s" or ch == "S") and i + 6 <= n:
            word = s[i:i+6]
            if word.lower() == "select":
                prev_ok = (i == 0) or (not is_word_char(s[i-1]))
                next_ok = (i + 6 == n) or (not is_word_char(s[i+6]))
                if prev_ok and next_ok:
                    return s[:i+6] + f" /*+LABEL('{label}')*/" + s[i+6:]

        i += 1

    # Fallback: prepend (still labels the statement in Vertica)
    return f"/*+LABEL('{label}')*/ " + sql


@mcp.tool()
async def profile_query(
    ctx: Context, query: str, timeout: Optional[int] = None
) -> dict:
    """Profile query execution with improved error handling."""
    await ctx.info("Profiling query")

    manager = ctx.request_context.lifespan_context.get("vertica_manager")
    if not manager:
        raise RuntimeError("No database connection manager available")

    conn = cursor = None
    label = f"mcp_profile_{uuid.uuid4().hex[:12]}"
    query_timeout = timeout or QUERY_TIMEOUT

    try:
        conn = manager.get_connection()
        cursor = conn.cursor()
        cursor.execute(f"SET SESSION RUNTIMECAP '{query_timeout}s'")

        labeled_sql = _inject_label(query, label)
        await ctx.debug(f"Executing with label: {label}")

        # Execute with PROFILE
        start_time = time.time()
        cursor.execute(f"PROFILE {labeled_sql}")
        execution_time = time.time() - start_time

        # Wait a moment for monitoring data to be recorded
        await asyncio.sleep(1)

        # Try multiple resolution strategies
        trxid = stmtid = duration_us = None
        
        # Strategy 1: query_profiles table
        cursor.execute("""
            SELECT transaction_id, statement_id, query_duration_us
            FROM v_monitor.query_profiles
            WHERE identifier = %s
            ORDER BY query_start_epoch DESC
            LIMIT 1
        """, (label,))
        
        row = cursor.fetchone()
        if row:
            trxid, stmtid, duration_us = row
            await ctx.info(f"Found in query_profiles: {trxid}-{stmtid}")
        else:
            # Strategy 2: query_requests table
            cursor.execute("""
                SELECT transaction_id, statement_id, request_duration_ms
                FROM v_monitor.query_requests
                WHERE request_label = %s
                ORDER BY start_timestamp DESC
                LIMIT 1
            """, (label,))
            
            row = cursor.fetchone()
            if row:
                trxid, stmtid, duration_ms = row
                duration_us = int(duration_ms) * 1000 if duration_ms else None
                await ctx.info(f"Found in query_requests: {trxid}-{stmtid}")

        if not trxid:
            # Strategy 3: Look for recent queries without label
            cursor.execute("""
                SELECT transaction_id, statement_id, query_duration_us
                FROM v_monitor.query_profiles
                WHERE query_start_epoch > %s
                ORDER BY query_start_epoch DESC
                LIMIT 5
            """, (start_time - 5,))  # 5 seconds before execution
            
            recent_queries = cursor.fetchall()
            if recent_queries:
                trxid, stmtid, duration_us = recent_queries[0]
                await ctx.warning(f"Using recent query (no label match): {trxid}-{stmtid}")

        if not trxid:
            # Fallback: return basic execution info
            return {
                "result": f"Query executed in {execution_time:.2f}s\nProfiling data not available",
                "query":labeled_sql,
                "label": label,
                "execution_time_seconds": execution_time,
                "note": "Could not resolve query IDs - profiling data unavailable"
            }

        # Get execution plan
        cursor.execute("""
            SELECT path_line
            FROM v_internal.dc_explain_plans
            WHERE transaction_id = %s 
            AND statement_id = %s
            ORDER BY path_id, path_line_index
        """, (trxid, stmtid))

        plan_rows = cursor.fetchall()
        plan_lines = [r[0] for r in plan_rows] if plan_rows else ["Plan not available"]

        result = f"Execution Time: {duration_us or int(execution_time * 1000000)}μs\n"
        result += f"Transaction ID: {trxid}\n"
        result += f"Statement ID: {stmtid}\n\n"
        result += "Execution Plan:\n"
        result += "\n".join(plan_lines)  # Limit plan lines

        return {
            "result": result,
            "query": query[:500],
            "label": label,
            "transaction_id": str(trxid),
            "statement_id": str(stmtid),
            "duration_us": duration_us,
            "plan_line_count": len(plan_lines),
        }

    except Exception as e:
        error_msg = f"Profile error: {str(e)}"
        await ctx.error(error_msg)
        raise RuntimeError(error_msg) from e
    finally:
        if cursor:
            cursor.close()
        if conn:
            manager.release_connection(conn)

#--------------------------------------------
#---------- Database Status ------------------
#--------------------------------------------
@mcp.tool()
async def database_status(ctx: Context) -> dict:
    """Get database status with improved error handling and formatting."""
    await ctx.info("Retrieving database status")

    manager = ctx.request_context.lifespan_context.get("vertica_manager")
    if not manager:
        raise RuntimeError("No database connection manager available")

    conn = cursor = None
    try:
        conn = manager.get_connection()
        cursor = conn.cursor()

        # Get version
        cursor.execute("SELECT version()")
        row = cursor.fetchone()
        version = row[0] if row else "Unknown"

        # Get current database size and usage (most recent audit)
        current_usage_query = """
        SELECT 
            (license_size_bytes / 1024^3)::NUMERIC(10, 2) AS license_gb,
            (database_size_bytes / 1024^3)::NUMERIC(10, 2) AS db_gb,
            (usage_percent * 100)::NUMERIC(5, 2) AS usage_pct,
            audit_start_timestamp,
            audit_end_timestamp
        FROM v_catalog.license_audits
        WHERE audit_end_timestamp = (
            SELECT MAX(audit_end_timestamp) 
            FROM v_catalog.license_audits
        )
        LIMIT 1;
        """

        cursor.execute(current_usage_query)
        current = cursor.fetchone()

        # Get usage trend (last 7 days)
        trend_query = """
        SELECT 
            DATE(audit_end_timestamp) as audit_date,
            AVG((usage_percent * 100))::NUMERIC(5, 2) AS avg_usage_pct,
            MAX((database_size_bytes / 1024^3))::NUMERIC(10, 2) AS max_db_gb
        FROM v_catalog.license_audits
        WHERE audit_end_timestamp >= CURRENT_DATE - INTERVAL '7 days'
        GROUP BY DATE(audit_end_timestamp)
        ORDER BY audit_date DESC
        LIMIT 7;
        """

        cursor.execute(trend_query)
        trend_data = cursor.fetchall()

        # Format results
        result = f"Database Status Report\n"
        result += f"Version: {version[:60]}\n\n"

        if current:
            license_gb, db_gb, usage_pct, start_time, end_time = current
            result += f"Current Usage:\n"
            result += f"- Database Size: {db_gb} GB\n"
            result += f"- License Capacity: {license_gb} GB\n"
            result += f"- Utilization: {usage_pct}% ({db_gb}/{license_gb} GB)\n"
            result += f"- Last Updated: {end_time}\n\n"
            
            # Add status indicator
            if usage_pct > 90:
                result += f"⚠️  Status: CRITICAL - Near capacity limit\n"
            elif usage_pct > 75:
                result += f"⚡ Status: WARNING - High utilization\n"
            else:
                result += f"✅ Status: HEALTHY - Normal utilization\n"
        else:
            result += "Current Usage: No audit data available\n"

        if trend_data:
            result += f"\n7-Day Usage Trend:\n"
            for date, avg_usage, max_db in trend_data:
                result += f"- {date}: {avg_usage}% ({max_db} GB)\n"

        # Get node count and cluster info
        cursor.execute("SELECT COUNT(*) FROM v_catalog.nodes WHERE node_state = 'UP'")
        node_count = cursor.fetchone()[0]
        result += f"\nCluster Info:\n"
        result += f"- Active Nodes: {node_count}\n"

        return {
            "result": result,
            "version": version,
            "current_usage_pct": float(current[2]) if current else 0,
            "current_db_size_gb": float(current[1]) if current else 0,
            "license_capacity_gb": float(current[0]) if current else 0,
            "trend_data_points": len(trend_data),
            "cluster_nodes": node_count
        }

    except Exception as e:
        error_msg = f"Error getting status: {str(e)}"
        await ctx.error(error_msg)
        raise RuntimeError(error_msg) from e
    finally:
        if cursor:
            cursor.close()
        if conn:
            manager.release_connection(conn)

#--------------------------------------------
#---------- System Performance ------------------
#--------------------------------------------
@mcp.tool()
async def analyze_system_performance(
    ctx: Context,
    window_minutes: int = 10,
    bucket: str = "minute",
    top_n: int = 5,
    flush: bool = True,
) -> dict:
    """Analyze system performance with improved efficiency."""
    await ctx.info(f"Analyzing performance (window={window_minutes}m)")

    manager = ctx.request_context.lifespan_context.get("vertica_manager")
    if not manager:
        raise RuntimeError("No database connection manager available")

    bucket = bucket.lower()
    if bucket not in {"second", "minute", "hour"}:
        raise ValueError("Invalid bucket value")

    def _rows_to_dicts(cur, rows):
        cols = [d[0] for d in cur.description] if cur.description else []
        return [dict(zip(cols, r)) for r in rows]

    conn = cursor = None
    try:
        conn = manager.get_connection()

        if not await _validate_connection(conn):
            manager.release_connection(conn)
            conn = manager.get_connection()

        cursor = conn.cursor()

        if flush:
            try:
                cursor.execute("SELECT FLUSH_DATA_COLLECTOR();")
                cursor.fetchall()
            except Exception:
                pass  # Non-critical

        window_minutes = max(1, int(window_minutes))
        top_n = max(1, min(10, int(top_n)))  # Cap at 10
        ts_expr = f"DATE_TRUNC('{bucket}', end_time)"
        where = f"end_time >= CURRENT_TIMESTAMP - INTERVAL '{window_minutes} minutes'"

        # CPU metrics
        cpu_sql = f"""
            SELECT node_name, {ts_expr} AS ts,
                   AVG(average_cpu_usage_percent) AS cpu_pct
            FROM v_monitor.cpu_usage
            WHERE {where}
            GROUP BY node_name, ts
            ORDER BY node_name, ts
            LIMIT 100
        """

        cursor.execute(cpu_sql)
        cpu_rows = _rows_to_dicts(cursor, cursor.fetchall())

        # Memory metrics
        mem_sql = f"""
            SELECT node_name, {ts_expr} AS ts,
                   AVG(average_memory_usage_percent) AS mem_pct
            FROM v_monitor.memory_usage
            WHERE {where}
            GROUP BY node_name, ts
            ORDER BY node_name, ts
            LIMIT 100
        """

        cursor.execute(mem_sql)
        mem_rows = _rows_to_dicts(cursor, cursor.fetchall())

        # Top tables by ROS
        top_tables_sql = f"""
            SELECT anchor_table_schema, anchor_table_name,
                   SUM(ros_count) AS total_ros_containers
            FROM v_monitor.projection_storage
            GROUP BY 1,2
            ORDER BY total_ros_containers DESC
            LIMIT {top_n}
        """

        cursor.execute(top_tables_sql)
        top_tables = _rows_to_dicts(cursor, cursor.fetchall())

        return {
            "cpu": cpu_rows[:50],  # Limit rows
            "memory": mem_rows[:50],
            "top_tables_by_ros": top_tables,
            "meta": {
                "window_minutes": window_minutes,
                "bucket": bucket,
                "top_n": top_n,
            },
        }

    except Exception as e:
        error_msg = f"Performance analysis error: {str(e)}"
        await ctx.error(error_msg)
        raise RuntimeError(error_msg) from e
    finally:
        if cursor:
            cursor.close()
        if conn:
            manager.release_connection(conn)
            

#--------------------------------------------
#---------- Generate Health Dashboard ------------------
#--------------------------------------------
@mcp.tool()
async def generate_health_dashboard(
    ctx: Context, output_format: str = "compact"
) -> dict:
    """Generate consolidated health dashboard with controlled output.
    Args:
        ctx: The context object.
        output_format: The format of the dashboard (default: compact, detailed, json).
    Returns:
        A dictionary containing the health dashboard.
    """

    manager = ctx.request_context.lifespan_context.get("vertica_manager")
    if not manager:
        raise RuntimeError("No database connection manager available")

    # Collect metrics efficiently
    try:
        status = await database_status(ctx)
        perf = await analyze_system_performance(ctx, window_minutes=5, top_n=3)
    except Exception as e:
        return {"error": f"Failed to collect metrics: {e}"}

    if output_format == "json":
        # Extract key metrics
        cpu_avg = _calculate_avg(perf.get("cpu", []), "cpu_pct")
        mem_avg = _calculate_avg(perf.get("memory", []), "mem_pct")

        return {
            "timestamp": datetime.now().isoformat(),
            "version": status.get("version", "Unknown")[:50],
            "metrics": {
                "cpu_pct": round(cpu_avg, 1),
                "memory_pct": round(mem_avg, 1),
            },
            "alerts": _generate_alerts(perf),
            "top_ros": [
                t["anchor_table_name"] for t in perf.get("top_tables_by_ros", [])
            ][:3],
        }

    elif output_format == "compact":
        return {
            "result": _format_compact_dashboard(status, perf),
            "token_estimate": 150,
        }

    else:  # detailed
        return {
            "result": _format_detailed_dashboard(status, perf),
            "token_estimate": 400,
        }


def _format_compact_dashboard(status: dict, perf: dict) -> str:
    """Ultra-compact dashboard."""
    cpu_avg = _calculate_avg(perf.get("cpu", []), "cpu_pct")
    mem_avg = _calculate_avg(perf.get("memory", []), "mem_pct")

    alerts = _generate_alerts(perf)
    alert_str = f"{len(alerts)} alerts" if alerts else "OK"

    return f"""Health Summary
CPU: {cpu_avg:.1f}% | Mem: {mem_avg:.1f}%
Status: {alert_str}"""


def _format_detailed_dashboard(status: dict, perf: dict) -> str:
    """Detailed but controlled dashboard."""
    try:
        cpu_avg = _calculate_avg(perf.get("cpu", []), "cpu_pct")
        mem_avg = _calculate_avg(perf.get("memory", []), "mem_pct")

        result = f"Database Health Report\n"
        result += f"Version: {status.get('version', 'Unknown')[:50]}\n\n"
        result += f"Resources:\n"
        result += f"- CPU: {cpu_avg:.1f}%\n"
        result += f"- Memory: {mem_avg:.1f}%\n\n"

        # Debug: Add performance data info
        result += f"Performance Data Keys: {list(perf.keys())}\n"
        result += f"CPU Records: {len(perf.get('cpu', []))}\n"
        result += f"Memory Records: {len(perf.get('memory', []))}\n\n"

        # Safe alert generation
        try:
            alerts = _generate_alerts(perf)
            if alerts:
                result += f"Alerts ({len(alerts)}):\n"
                for alert in alerts[:3]:
                    result += f"- {alert.get('type', 'Unknown')}: {alert.get('value', 'N/A')}\n"
            else:
                result += "Alerts: None\n"
        except Exception as e:
            result += f"Alert Generation Error: {str(e)}\n"

        # Add top tables info
        try:
            top_tables = perf.get("top_tables_by_ros", [])
            if top_tables:
                result += f"\nTop ROS Tables:\n"
                for table in top_tables[:3]:
                    name = table.get("anchor_table_name", "Unknown")
                    ros_count = table.get("total_ros_containers", 0)
                    result += f"- {name}: {ros_count} containers\n"
            else:
                result += "\nTop Tables: No data available\n"
        except Exception as e:
            result += f"\nTop Tables Error: {str(e)}\n"

        return result

    except Exception as e:
        return f"Dashboard Generation Error: {str(e)}\nRaw Status: {status}\nRaw Perf: {perf}"


def _calculate_avg(data: list, field: str) -> float:
    """Calculate field average."""
    if not data:
        return 0.0
    values = [float(d.get(field, 0)) for d in data if field in d]
    return sum(values) / len(values) if values else 0.0


def _generate_alerts(perf: dict) -> list:
    """Generate performance alerts with better error handling."""
    alerts = []
    
    try:
        # CPU alerts
        cpu_data = perf.get("cpu", [])
        if cpu_data:
            cpu_avg = _calculate_avg(cpu_data, "cpu_pct")
            if cpu_avg > 85:
                alerts.append({"type": "cpu_high", "value": f"{cpu_avg:.1f}%"})

        # Memory alerts  
        mem_data = perf.get("memory", [])
        if mem_data:
            mem_avg = _calculate_avg(mem_data, "mem_pct")
            if mem_avg > 85:
                alerts.append({"type": "memory_high", "value": f"{mem_avg:.1f}%"})

        # ROS container alerts
        top_tables = perf.get("top_tables_by_ros", [])
        for table in top_tables:
            ros_count = table.get("total_ros_containers", 0)
            if ros_count > 5000:
                alerts.append({
                    "type": "ros_high",
                    "table": table.get("anchor_table_name", "Unknown"),
                    "value": f"{ros_count:,} containers"
                })

    except Exception as e:
        alerts.append({"type": "error", "value": f"Alert generation failed: {str(e)}"})

    return alerts


#--------------------------------------------
#---------- Health Dashboard Prompt ------------------
#--------------------------------------------
@mcp.prompt()
async def vertica_database_health_dashboard() -> str:
    """Compact health dashboard."""
    return """HEALTH DASHBOARD
Call: generate_health_dashboard(format="compact")
Show: Version, usage%, alerts only"""


#--------------------------------------------
#---------- System Monitor Prompt ------------------
#--------------------------------------------
@mcp.prompt()
async def vertica_database_system_monitor() -> str:
    """System performance monitor."""
    return """SYSTEM MONITOR
Call: analyze_system_performance(window_minutes=5, top_n=3)
Show: CPU/Mem avg, top ROS tables
Alert if >85% usage or >5000 ROS"""



#--------------------------------------------
#---------- Compact Health Report Prompt ------------------
#--------------------------------------------
@mcp.prompt()
async def vertica_compact_health_report() -> str:
    """Token-efficient health report."""
    return """COMPACT REPORT
Call: generate_health_dashboard(format="json")
Return JSON with summary + alerts only
Max 300 tokens"""

#--------------------------------------------
#---------- SQL Safety Guard Prompt ------------------
#--------------------------------------------
@mcp.prompt()
async def sql_query_safety_guard() -> str:
    """SQL safety guard."""
    return """SQL SAFETY
1. Call run_query_safely with row_threshold=1000
2. If large, ask user to confirm
3. Use pagination for results"""


#--------------------------------------------
#---------- Query Performance Analyzer Prompt ------------------
#--------------------------------------------
@mcp.prompt()
async def vertica_query_performance_analyzer() -> str:
    """
    🚀 Vertica Performance Analyzer - Deep-dive query performance analysis with actionable optimization recommendations.

    Analyzes the given query execution plans, identifies bottlenecks, and provides concrete DDL suggestions
    for optimal Vertica projections, join strategies, and ROS container health.
    """
    return """🔍 VERTICA PERFORMANCE ANALYSIS

ANALYSIS WORKFLOW:
1) IMPORTANT: ALWAYS SUGGEST TO THE USER TO USE DATABASE DESIGNER (DBD) COMPREHENSIVE AND INCREMENTAL TO OPTIMIZE THE QUERY FIRST
2) **PROFILE EXECUTION**: Call `profile_query` to get actual runtime and execution plan
3) **PARSE OPERATORS**: Identify all operators (Scan, Hash Join, Merge Join, GroupBy, Sort, Exchange)
4) **COST ANALYSIS**: Find the 3-5 highest-cost operations with row estimates
5) **JOIN OPTIMIZATION**: For each Hash Join → propose Merge Join strategy
6) **AGGREGATION TUNING**: For each Hash GroupBy → propose Pipelined GroupBy
7) **PROJECTION HEALTH**: Audit ROS container counts for projections used
8) **CONCRETE RECOMMENDATIONS**: Specific DDL for optimal projections

OPTIMIZATION FOCUS:
- **Merge Joins**: Require pre-sorted data (ORDER BY) and co-location (SEGMENTED BY HASH)
- **Pipelined GroupBy**: Needs input sorted on GROUP BY columns
- **ROS Container Health**: Flag projections with >5000 containers for maintenance

OUTPUT SECTIONS:
1. **Executive Summary**: Duration, row estimates, performance grade
2. **Top Cost Operations**: Operator | Input | Rows | Cost/Notes
3. **Join Optimization**: Current → Target with specific ORDER BY/SEGMENTED BY
4. **GroupBy Optimization**: Hash → Pipelined with projection requirements  
5. **Projection Health**: Base Projection | ROS Containers | Status
6. **Action Items**: Concrete DDL statements and maintenance tasks

RULES:
- Start with `profile_query` - no exceptions
- Use `execute_query_paginated` for large result sets, never full table scans
- Normalize projection names (strip _b0/_b1 suffixes)
- Be specific: exact column lists, not generic advice
- Flag urgent issues: >5000 ROS containers, inefficient operators"""


#--------------------------------------------
#---------- SQL Assistant Prompt ------------------
#--------------------------------------------
@mcp.prompt()
async def vertica_sql_assistant() -> str:
    """
    💡 Vertica SQL Assistant - Expert SQL query generation with Vertica-specific optimizations.

    Generates efficient, Vertica-optimized SQL queries for any data task with proper
    function usage, performance considerations, and best practices.
    """
    return """📝 VERTICA SQL QUERY GENERATION

As a Vertica SQL expert, I'll write an optimized query for the given task.

VERTICA BEST PRACTICES:
✅ **Functions**: Use Vertica-specific functions (REGEXP_REPLACE, REGEXP_LIKE, APPROXIMATE_COUNT_DISTINCT, etc.)
✅ **Joins**: Prefer INNER joins when possible, use appropriate join types
✅ **Filtering**: Apply WHERE clauses early to minimize data movement
✅ **Aggregations**: Use efficient aggregation functions and GROUP BY strategies
✅ **Projections**: Consider projection-friendly query patterns
✅ **Formatting**: Clear, readable SQL with proper indentation

APPROACH:
1. If schema exploration needed → use `get_schema_tables`, `get_database_schemas`, `get_table_structure`
2. Write efficient, well-formatted SQL
3. Include comments explaining Vertica-specific optimizations
4. Suggest projection improvements if applicable

QUERY REQUIREMENTS:
- Readable formatting with clear indentation
- Proper use of Vertica functions and syntax
- Performance-conscious design
- Comments for complex logic
RULE: CHECK THE DOCUMENTATION FOR THE FUNCTIONS FIRST BEFORE USING THEM"""
