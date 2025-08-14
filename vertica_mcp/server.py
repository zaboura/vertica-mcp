"""Server module for Vertica MCP. Provides API endpoints and database utilities."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Any
import logging
import re
import os
import socket
import uuid
from mcp.server.fastmcp import FastMCP, Context
from dotenv import load_dotenv, find_dotenv
import uvicorn
from vertica_mcp.connection import (
    VerticaConnectionManager,
    VerticaConfig,
    OperationType,
)


MCP_SERVER_NAME = "vertica-mcp"

DEPENDENCIES = ["vertica-python", "pydantic", "starlette", "uvicorn"]
# Configure logging
logger = logging.getLogger("vertica-mcp")


def _is_select(query: str) -> bool:
    """Check if a query is a SELECT statement."""
    q = query.strip()
    # allow wrapping parens
    while q.startswith("(") and q.endswith(")"):
        q = q[1:-1].strip()
    return q.upper().startswith("SELECT")


def _wrap_subquery(sql: str) -> str:
    """Wrap a subquery in a SELECT statement."""
    sql = sql.replace(";", "").strip()
    return f"SELECT * FROM ({sql}) q"


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
    """Extract schema name from a SQL query."""
    query = query.strip().lower()
    match = re.search(r"([a-zA-Z0-9_]+)\.[a-zA-Z0-9_]+", query)
    if match:
        return match.group(1)
    return "public"  # Default schema if none found


async def run_sse(host: str = "localhost", port: int = 8000) -> None:
    """Launch the MCP server with HTTP-SSE transport."""
    logger.info(f"Starting MCP server with SSE transport on {host}:{port}")

    # Get the SSE app directly from FastMCP
    # This app already has all the routes it needs
    sse_app = mcp.sse_app()

    # Print startup information
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
    print(f"   Or use MCP Inspector: mcp dev vertica_mcp/server.py\n")

    # Run the SSE app directly without additional mounting
    config = uvicorn.Config(
        sse_app,  # Use the SSE app directly
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        use_colors=True,
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
    """Server lifespan context manager that handles initialization and cleanup."""

    # ROBUST .env loading - try multiple methods
    env_loaded = False

    # Method 1: Try find_dotenv with usecwd=True
    try:
        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            load_dotenv(dotenv_path, override=False)
            logger.info("Loaded environment from %s", dotenv_path)
            env_loaded = True
    except Exception as e:
        logger.debug("find_dotenv(usecwd=True) failed: %s", e)

    # Method 2: Try current working directory
    if not env_loaded:
        try:
            cwd_env = os.path.join(os.getcwd(), ".env")
            if os.path.exists(cwd_env):
                load_dotenv(cwd_env, override=False)
                logger.info("Loaded environment from current directory: %s", cwd_env)
                env_loaded = True
        except Exception as e:
            logger.debug("Current directory .env failed: %s", e)

    # Method 3: Try relative to script location
    if not env_loaded:
        try:
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            script_env = os.path.join(script_dir, ".env")
            if os.path.exists(script_env):
                load_dotenv(script_env, override=False)
                logger.info("Loaded environment from script directory: %s", script_env)
                env_loaded = True
        except Exception as e:
            logger.debug("Script directory .env failed: %s", e)

    # Method 4: Just try default load_dotenv()
    if not env_loaded:
        try:
            load_dotenv()
            logger.info("Loaded environment using default load_dotenv()")
            env_loaded = True
        except Exception as e:
            logger.debug("Default load_dotenv() failed: %s", e)

    if not env_loaded:
        logger.warning("No .env file loaded - using system environment variables only")

    manager = None
    try:
        manager = VerticaConnectionManager()
        config = VerticaConfig.from_env()
        logger.info(
            "Vertica cfg → host=%s port=%d db=%s user=%s ssl=%s",
            config.host,
            config.port,
            config.database,
            config.user,
            config.ssl,
        )

        # DEBUG: Test basic connectivity first
        try:
            logger.info(
                f"Testing basic TCP connectivity to {config.host}:{config.port}..."
            )
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            result = sock.connect_ex((config.host, config.port))
            sock.close()
            if result == 0:
                logger.info("✅ TCP connectivity successful")
            else:
                logger.error(f"❌ TCP connectivity failed with error code: {result}")
        except Exception as sock_error:
            logger.error(f"❌ Socket test failed: {sock_error}")

        try:
            manager.initialize_default(config)
            logger.info("Vertica connection manager initialized")

            # Test the connection immediately
            logger.info("Testing Vertica database connection...")
            conn = None
            try:
                conn = manager.get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT version()")
                version = cursor.fetchone()[0]
                logger.info(f"✅ Successfully connected to Vertica: {version}")
                cursor.close()
            except Exception as conn_error:
                logger.error(f"❌ Failed to test Vertica connection: {conn_error}")
                logger.error(f"❌ Connection error type: {type(conn_error).__name__}")
                logger.error(f"❌ Connection error details: {str(conn_error)}")
                logger.warning("Server will continue but database operations will fail")
            finally:
                if conn:
                    manager.release_connection(conn)

        except Exception as e:
            logger.error("❌ DB init failed at startup (continuing): %s", str(e))
            logger.error(f"❌ DB init error type: {type(e).__name__}")

        yield {"vertica_manager": manager}

    except Exception as e:
        logger.error("❌ Failed to initialize server (fatal): %s", str(e))
        logger.error(f"❌ Server init error type: {type(e).__name__}")
        yield {"vertica_manager": manager}
    finally:
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


@mcp.tool()
async def run_query_safely(
    ctx: Context,
    query: str,
    row_threshold: int = 1000,
    proceed: bool = False,
    mode: str = "page",  # "page" (recommended) or "stream"
    page_limit: int = 2000,  # first page size when proceeding
    include_columns: bool = True,
    precount: bool = False,  # if True and "large", also do COUNT(*) (costly)
) -> dict:
    """
    Gatekeeper for user queries:
    - If non-SELECT: execute immediately.
    - If SELECT and not proceeding: quick probe using LIMIT to detect "large".
      -> emits ctx.warning and returns requires_confirmation + preview.
    - If proceeding: fetch using paging (or stream if you really want it).
    """
    await ctx.info("run_query_safely called")

    manager = ctx.request_context.lifespan_context.get("vertica_manager")
    if not manager:
        await ctx.error("No database connection manager available")
        raise RuntimeError("No database connection manager available")

    # Permission check for DDL/DML
    schema = extract_schema_from_query(query)
    operation = extract_operation_type(query)
    if operation and not manager.is_operation_allowed(schema or "default", operation):
        msg = f"Operation {operation.name} not allowed for schema {schema}"
        await ctx.error(msg)
        raise RuntimeError(msg)

    # Non-SELECT -> just run it
    if not _is_select(query):
        conn = cursor = None
        try:
            conn = manager.get_connection()
            cursor = conn.cursor()
            cursor.execute(query)
            affected = getattr(cursor, "rowcount", None)
            await ctx.info(f"Non-SELECT executed, affected_rows={affected}")
            return {"ok": True, "affected_rows": affected}
        except Exception as e:
            msg = f"Error executing statement: {e}"
            await ctx.error(msg)
            raise RuntimeError(msg) from e
        finally:
            if cursor:
                cursor.close()
            if conn:
                manager.release_connection(conn)

    # SELECT flow
    if not proceed:
        # Probe with LIMIT (cheap). If > threshold, warn and ask.
        probe_limit = row_threshold + 1
        probe_sql = f"{_wrap_subquery(query)} LIMIT {probe_limit}"

        conn = cursor = None
        try:
            conn = manager.get_connection()
            cursor = conn.cursor()
            cursor.execute(probe_sql)
            rows = cursor.fetchall()
            cols = (
                [d[0] for d in cursor.description]
                if (include_columns and cursor.description)
                else None
            )

            is_large = len(rows) > row_threshold
            preview = rows[: min(50, len(rows))]  # keep preview tiny

            exact_count = None
            if is_large and precount:
                await ctx.info(
                    "Computing exact COUNT(*) for large result (may be expensive)"
                )
                cursor.execute(f"SELECT COUNT(*) FROM ({query}) q")
                exact_count = int(cursor.fetchone()[0])

            if not is_large:
                await ctx.info(
                    f"Small result (<= {row_threshold}). Returning immediately."
                )
                return {
                    "ok": True,
                    "rows": rows,
                    "count": len(rows),
                    "done": True,
                    "columns": cols,
                    "large": False,
                }

            # Large -> warn & require confirmation
            human_msg = (
                f"Large result detected (> {row_threshold} rows)"
                + (f": about {exact_count} rows." if exact_count is not None else ".")
                + " Proceed?"
            )
            await ctx.warning(human_msg)  # shows yellow warning in Claude Desktop

            # Hand the client an explicit next step to call if they accept
            return {
                "ok": True,
                "large": True,
                "requires_confirmation": True,
                "threshold": row_threshold,
                "exact_count": exact_count,  # may be None if precount=False
                "message": human_msg,
                "preview": preview,  # tiny peek
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

    # proceed=True
    await ctx.info(f"Proceeding with mode={mode}")

    if mode == "page":
        # Reuse your paging implementation
        return await execute_query_paginated(
            ctx=ctx,
            query=query,
            limit=page_limit,
            offset=0,
            include_columns=include_columns,
        )

    if mode == "stream":
        # Streaming returns a giant JSON in your current impl – not ideal.
        # If you insist, call it; otherwise prefer paging.
        return await execute_query_stream(
            ctx=ctx, query=query, batch_size=max(page_limit, 1000)
        )

    raise RuntimeError(f"Unknown mode: {mode}")


@mcp.tool()
async def execute_query_paginated(
    ctx: Context,
    query: str,
    limit: int = 2000,
    offset: int = 0,
    include_columns: bool = True,
) -> dict:
    """
    Execute a SELECT query with LIMIT/OFFSET paging.

    Returns:
      {
        "rows": List[List[Any]],
        "count": int,              # number of rows in this page
        "next_offset": int,        # where to start the next page
        "done": bool,              # True if no more rows
        "columns": List[str]       # optional, only when include_columns=True
      }
    """
    await ctx.info(f"execute_query_paginated(limit={limit}, offset={offset})")
    manager = ctx.request_context.lifespan_context.get("vertica_manager")
    if not manager:
        await ctx.error("No database connection manager available")
        raise RuntimeError("No database connection manager available")

    # Basic safety: SELECT-only
    op = extract_operation_type(query)
    if op:
        raise RuntimeError("execute_query_paginated only supports SELECT statements")

    # Wrap the user query to safely apply LIMIT/OFFSET
    paged_sql = f"{_wrap_subquery(query)} LIMIT {int(limit)} OFFSET {int(offset)}"

    conn = None
    cursor = None
    try:
        conn = manager.get_connection()
        cursor = conn.cursor()
        cursor.execute(paged_sql)
        rows = cursor.fetchall()
        cols = (
            [d[0] for d in cursor.description]
            if (include_columns and cursor.description)
            else None
        )

        done = len(rows) < limit
        resp = {
            "rows": rows,
            "count": len(rows),
            "next_offset": offset + len(rows),
            "done": done,
        }
        if cols is not None:
            resp["columns"] = cols
        return resp
    except Exception as e:
        msg = f"Error in execute_query_paginated: {str(e)}"
        await ctx.error(msg)
        raise RuntimeError(msg) from e
    finally:
        if cursor:
            cursor.close()
        if conn:
            manager.release_connection(conn)


@mcp.tool()
async def execute_query_stream(
    ctx: Context, query: str, batch_size: int = 1000
) -> dict:
    """Execute a SQL query and return results in batches with pagination support.

    Args:
        ctx (Context): FastMCP context for progress reporting and logging.
        query (str): SQL query to execute.
        batch_size (int, optional): Number of rows to fetch per batch (default: 1000).

    Returns:
        dict: Dictionary containing results, total number of rows, and truncation information.
        If an error occurs, returns an error message.
    """
    await ctx.info(f"Executing query with batch size: {batch_size}")
    manager = ctx.request_context.lifespan_context.get("vertica_manager")
    if not manager:
        await ctx.error("No database connection manager available")
        raise RuntimeError("No database connection manager available")
    schema = extract_schema_from_query(query)
    operation = extract_operation_type(query)
    if operation and not manager.is_operation_allowed(schema or "default", operation):
        error_msg = f"Operation {operation.name} not allowed for schema {schema}"
        await ctx.error(error_msg)
        raise RuntimeError(error_msg)
    conn = None
    cursor = None
    try:
        conn = manager.get_connection()
        cursor = conn.cursor()
        cursor.execute(query)
        all_results = []
        total_rows = 0
        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break
            total_rows += len(batch)
            all_results.extend(batch)
            await ctx.debug(f"Fetched {total_rows} rows")
            if total_rows > 100000:
                await ctx.warning("Result set too large, truncating at 100,000 rows")
                break
        await ctx.info(f"Query completed, total rows: {total_rows}")
        return {
            "result": all_results,
            "total_rows": total_rows,
            "truncated": total_rows > 100000,
        }
    except Exception as e:
        error_msg = f"Error executing query: {str(e)}"
        await ctx.error(error_msg)
        raise RuntimeError(error_msg) from e
    finally:
        if cursor:
            cursor.close()
        if conn:
            manager.release_connection(conn)


@mcp.tool()
async def get_table_structure(
    ctx: Context, table_name: str, schema_name: str = "public"
) -> dict:
    """Get the structure of a Vertica table, including columns, data types, and constraints.

    Args:
        ctx (Context): FastMCP context for progress reporting and logging.
        table_name (str): Name of the table to inspect.
        schema_name (str, optional): Schema name (default: "public").

    Returns:
        dict: Dictionary containing table structure information.
        If an error occurs, returns an error message.
    """
    await ctx.info(f"Getting structure for table: {schema_name}.{table_name}")
    manager = ctx.request_context.lifespan_context.get("vertica_manager")
    if not manager:
        await ctx.error("No database connection manager available")
        raise RuntimeError("No database connection manager available")
    query = """
    SELECT
        column_name,
        data_type,
        character_maximum_length,
        numeric_precision,
        numeric_scale,
        is_nullable,
        column_default
    FROM v_catalog.columns
    WHERE table_schema = %s
    AND table_name = %s
    ORDER BY ordinal_position;
    """
    conn = None
    cursor = None
    try:
        conn = manager.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, (schema_name, table_name))
        columns = cursor.fetchall()
        if not columns:
            raise RuntimeError(f"No table found: {schema_name}.{table_name}")
        cursor.execute(
            """
            SELECT
                constraint_name,
                constraint_type,
                column_name
            FROM v_catalog.constraint_columns
            WHERE table_schema = %s
            AND table_name = %s;
        """,
            (schema_name, table_name),
        )
        constraints = cursor.fetchall()
        result = f"Table Structure for {schema_name}.{table_name}:\n\n"
        result += "Columns:\n"
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
        return {
            "result": result,
            "table_name": table_name,
            "schema": schema_name,
            "column_count": len(columns),
            "constraint_count": len(constraints),
        }
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
    """List all projections for a specific Vertica table.

    Args:
        ctx (Context): FastMCP context for progress reporting and logging.
        table_name (str): Name of the table to inspect.
        schema_name (str, optional): Schema name (default: "public").

    Returns:
        dict: Dictionary containing projections information.
        If an error occurs, returns an error message.
    """
    await ctx.info(f"Listing projections for table: {schema_name}.{table_name}")
    manager = ctx.request_context.lifespan_context.get("vertica_manager")
    if not manager:
        await ctx.error("No database connection manager available")
        raise RuntimeError("No database connection manager available")
    query = """
    SELECT
        projection_name,
        is_super_projection,
        anchor_table_name,
        create_type
    FROM v_catalog.projections
    WHERE projection_schema = %s
    AND anchor_table_name = %s
    ORDER BY projection_name;
    """
    conn = None
    cursor = None
    try:
        conn = manager.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, (schema_name, table_name))
        projections = cursor.fetchall()
        if not projections:
            raise RuntimeError(
                f"No projections found for table: {schema_name}.{table_name}"
            )
        result = f"Projections for {schema_name}.{table_name}:\n\n"
        for proj in projections:
            result += f"- {proj[0]} (Super Projection: {proj[1]}) [Table: {proj[2]}] (Creation Type: {proj[3]})\n"
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


@mcp.tool()
async def get_schema_views(ctx: Context, schema_name: str = "public") -> dict:
    """List all views in a specific Vertica schema.

    Args:
        ctx (Context): FastMCP context for progress reporting and logging.
        schema_name (str, optional): Schema name (default: "public").

    Returns:
        dict: Dictionary containing views information.
        If an error occurs, returns an error message.
    """
    await ctx.info(f"Listing views in schema: {schema_name}")
    manager = ctx.request_context.lifespan_context.get("vertica_manager")
    if not manager:
        await ctx.error("No database connection manager available")
        raise RuntimeError("No database connection manager available")
    query = """
    SELECT
        table_name,
        view_definition
    FROM v_catalog.views
    WHERE table_schema = %s
    ORDER BY table_name;
    """
    conn = None
    cursor = None
    try:
        conn = manager.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, (schema_name,))
        views = cursor.fetchall()
        if not views:
            raise RuntimeError(f"No views found in schema: {schema_name}")
        result = f"Views in schema {schema_name}:\n\n"
        for view in views:
            result += f"View: {view[0]}\n"
            result += f"Definition:\n{view[1]}\n\n"
        return {"result": result, "schema": schema_name, "view_count": len(views)}
    except Exception as e:
        error_msg = f"Error listing views: {str(e)}"
        await ctx.error(error_msg)
        raise RuntimeError(error_msg) from e
    finally:
        if cursor:
            cursor.close()
        if conn:
            manager.release_connection(conn)


@mcp.tool()
async def get_schema_tables(ctx: Context, schema_name: str = "public") -> dict:
    """List all tables in a specific Vertica schema.

    Args:
        ctx (Context): FastMCP context for progress reporting and logging.
        schema_name (str, optional): Schema name (default: "public").

    Returns:
        dict: Dictionary containing tables information.
        If an error occurs, returns an error message.
    """
    await ctx.info(f"Listing tables in schema: {schema_name}")
    manager = ctx.request_context.lifespan_context.get("vertica_manager")
    if not manager:
        await ctx.error("No database connection manager available")
        raise RuntimeError("No database connection manager available")
    query = """
    SELECT
        table_name
    FROM v_catalog.tables
    WHERE table_schema = %s
    ORDER BY table_name;
    """
    conn = None
    cursor = None
    try:
        conn = manager.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, (schema_name,))
        tables = cursor.fetchall()
        if not tables:
            raise RuntimeError(f"No tables found in schema: {schema_name}")
        result = f"Tables in schema {schema_name}:\n\n"
        for table in tables:
            result += f"{table[0]}\n"
        return {"result": result, "schema": schema_name, "table_count": len(tables)}
    except Exception as e:
        error_msg = f"Error listing tables: {str(e)}"
        await ctx.error(error_msg)
        raise RuntimeError(error_msg) from e
    finally:
        if cursor:
            cursor.close()
        if conn:
            manager.release_connection(conn)


@mcp.tool()
async def get_database_schemas(ctx: Context) -> dict:
    """List all schemas in the Vertica database.

    Args:
        ctx (Context): FastMCP context for progress reporting and logging.

    Returns:
        dict: Dictionary containing schemas information.
        If an error occurs, returns an error message.
    """
    await ctx.info("Listing schemas in the Vertica database.")
    manager = ctx.request_context.lifespan_context.get("vertica_manager")
    if not manager:
        await ctx.error("No database connection manager available")
        raise RuntimeError("No database connection manager available")
    query = """
    SELECT
        schema_name,
        is_system_schema
    FROM v_catalog.schemata
    ORDER BY schema_name;
    """
    conn = None
    cursor = None
    try:
        conn = manager.get_connection()
        cursor = conn.cursor()
        cursor.execute(query)
        schemas = cursor.fetchall()
        if not schemas:
            raise RuntimeError("No schemas found.")
        result = "Schemas in the Vertica database:\n\n"
        for schema in schemas:
            result += f"{schema[0]}\n"
        return {"result": result, "schema_count": len(schemas)}
    except Exception as e:
        error_msg = f"Error listing schemas: {str(e)}"
        await ctx.error(error_msg)
        raise RuntimeError(error_msg) from e
    finally:
        if cursor:
            cursor.close()
        if conn:
            manager.release_connection(conn)


@mcp.tool()
async def database_status(ctx: Context) -> dict:
    """Retrieve database status including usage statistics and version information.

    Args:
        ctx (Context): FastMCP context for progress reporting and logging.

    Returns:
        dict: Dictionary containing database status information.
        If an error occurs, returns an error message.
    """
    await ctx.info("Retrieving database status information.")
    manager = ctx.request_context.lifespan_context.get("vertica_manager")
    if not manager:
        await ctx.error("No database connection manager available")
        raise RuntimeError("No database connection manager available")

    conn = None
    cursor = None
    try:
        conn = manager.get_connection()
        cursor = conn.cursor()

        # Get Vertica version
        cursor.execute("SELECT version()")
        version_result = cursor.fetchone()
        version = version_result[0] if version_result else "Unknown"

        # Get usage statistics
        usage_query = """
        SELECT * FROM
        ( SELECT
             (license_size_bytes / 1024 ^3)::NUMERIC(10, 2) AS license_size_GB
            , (database_size_bytes / 1024 ^3)::NUMERIC(10, 2) AS database_size_GB
            , (usage_percent * 100)::NUMERIC(3, 1) AS usage_percent
            , audit_end_timestamp
            , audited_data
        FROM v_catalog.license_audits
        ORDER BY audit_end_timestamp DESC LIMIT 4) foo
        ORDER BY 5;
        """
        cursor.execute(usage_query)
        usages = cursor.fetchall()

        # Build result string
        result = f"Vertica Version: {version}\n\n"

        if usages:
            headers = [
                "License Size (GB)",
                "Database Size (GB)",
                "Usage (%)",
                "Audit End Timestamp",
                "Audited Data",
            ]
            result += "Database Usage Statistics:\n\n"
            result += " | ".join(headers) + "\n"
            result += "---------|---------|---------|----------|-----------\n"
            for row in usages:
                formatted_row = [str(val) if val is not None else "" for val in row]
                result += " | ".join(formatted_row) + "\n"
        else:
            result += "No usage statistics found."

        return {
            "result": result,
            "version": version,
            "record_count": len(usages) if usages else 0,
        }
    except Exception as e:
        error_msg = f"Error retrieving database status: {str(e)}"
        await ctx.error(error_msg)
        raise RuntimeError(error_msg) from e
    finally:
        if cursor:
            cursor.close()
        if conn:
            manager.release_connection(conn)


@mcp.tool()
async def profile_query(ctx: Context, query: str) -> dict:
    """
    Profile a SQL query and return the plan + duration.
    Fixes the bug where transaction_id/statement_id came from the *next* statement
    by labeling the profiled statement and then resolving IDs via system tables.
    """

    await ctx.info(f"Profiling query: {query}")

    manager = ctx.request_context.lifespan_context.get("vertica_manager")
    if not manager:
        await ctx.error("No database connection manager available")
        raise RuntimeError("No database connection manager available")

    def _inject_label(sql: str, label: str) -> str:
        # if already labeled, leave as-is
        if re.search(r"/\*\+\s*label\s*\(", sql, flags=re.IGNORECASE):
            logger.info("SQL already labeled, leaving as-is")
            return sql
        # statement keywords that support LABEL (per Vertica docs)
        stmt_keywords = [
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "MERGE",
            "COPY",
            "EXPORT",  # EXPORT TO ...
        ]
        # try to place after the first top-level SELECT/keyword
        for kw in stmt_keywords:
            pattern = re.compile(rf"^\s*({kw})\b", re.IGNORECASE)
            if pattern.search(sql):
                result = pattern.sub(
                    lambda m: f"{m.group(1)} /*+ LABEL('{label}') */", sql, count=1
                )
                logger.info(f"Labeled SQL with {kw}: {result}")
                return result
        # handle WITH (...) SELECT ... : put label at the first SELECT occurrence
        result = re.sub(
            r"\bSELECT\b",
            f"SELECT /*+ LABEL('{label}') */",
            sql,
            count=1,
            flags=re.IGNORECASE,
        )
        logger.info(f"Final labeled SQL: {result}")
        return result

    conn = None
    cursor = None
    label = f"mcp_profile_{uuid.uuid4().hex[:12]}"

    try:
        conn = manager.get_connection()
        cursor = conn.cursor()

        labeled_sql = _inject_label(query, label)
        await ctx.debug(f"Using label: {label}")
        await ctx.debug(f"Labeled SQL: {labeled_sql}")

        # Execute PROFILE on the labeled statement
        cursor.execute(f"PROFILE {labeled_sql}")

        # Resolve tx/stmt using the label from system tables
        cursor.execute(
            """
            SELECT transaction_id, statement_id, query_duration_us
            FROM v_monitor.query_profiles
            WHERE identifier = %s
            ORDER BY query_start_epoch DESC
            LIMIT 1
            """,
            (label,),
        )
        row = cursor.fetchone()

        trxid = stmtid = None
        duration_us = None

        if row:
            trxid, stmtid, duration_us = row
        else:
            logger.info("Not found in query_profiles, trying query_requests...")
            # Fallback: look in QUERY_REQUESTS by request_label
            cursor.execute(
                """
                SELECT transaction_id, statement_id, request_duration_ms
                FROM v_monitor.query_requests
                WHERE request_label = %s
                ORDER BY start_timestamp DESC
                LIMIT 1
                """,
                (label,),
            )
            row2 = cursor.fetchone()
            if row2:
                trxid, stmtid, request_duration_ms = row2
                duration_us = (
                    int(request_duration_ms) * 1000
                    if request_duration_ms is not None
                    else None
                )

        if trxid is None or stmtid is None:
            raise RuntimeError(
                "Could not resolve transaction_id/statement_id for profiled query"
            )

        # Fetch plan lines (to V_INTERNAL.DC_EXPLAIN_PLANS)
        cursor.execute(
            """
            SELECT path_line 
            FROM 
                v_internal.dc_explain_plans
            WHERE 
                transaction_id = %s
                AND statement_id = %s
            ORDER BY 
                   path_id,
                   path_line_index;
            """,
            (trxid, stmtid),
        )
        plan_rows = cursor.fetchall()

        plan_lines = [r[0] for r in (plan_rows or [])]
        plan_str = "\n".join(plan_lines) if plan_lines else "Query plan not found."

        # Duration string for human readability
        duration_str = (
            f"Execution Duration (us): {duration_us}"
            if duration_us is not None
            else "Duration not found."
        )
        result_text = f"{duration_str}\n\nQuery Plan:\n{plan_str}"

        await ctx.info("Profile completed.")
        return {
            "result": result_text,
            "query": query,
            "label": label,
            "transaction_id": trxid,
            "statement_id": stmtid,
            "duration_us": duration_us,
            "plan_line_count": len(plan_lines),
        }
    except Exception as e:
        error_msg = f"Error profiling query: {str(e)}"
        await ctx.error(error_msg)
        raise RuntimeError(error_msg) from e
    finally:
        if cursor:
            cursor.close()
        if conn:
            manager.release_connection(conn)


@mcp.tool()
async def analyze_system_performance(
    ctx: Context,
    window_minutes: int = 15,
    bucket: str = "minute",  # "second" | "minute" | "hour"
    limit_per_node: int | None = None,
    top_n: int = 5,
    flush: bool = True,  # run FLUSH_DATA_COLLECTOR() first
) -> dict:
    """
    Collects system performance metrics and hotspots.
    Args:
        window_minutes: The time window in minutes to collect metrics for.
        bucket: The time bucket to group metrics by.
        limit_per_node: The maximum number of nodes to collect metrics for.
        top_n: The number of top tables and projections to collect.
        flush: Whether to run FLUSH_DATA_COLLECTOR() first.
    Returns:
      timeseries:
        - cpu:    [{node_name, ts, cpu_pct}]
        - memory: [{node_name, ts, mem_pct}]
        - network:[{node_name, ts, tx_kbytes_per_sec, rx_kbytes_per_sec}]
      snapshots:
        - resource_pools: rows from v_monitor.resource_pool_status
        - top_tables_by_ros: [{anchor_table_schema, anchor_table_name, total_ros_containers, used_bytes}]
        - top_projections_by_ros: [{projection_schema, base_projection_name, total_ros_containers, used_bytes}]
          NOTE: _b0/_b1 copies are *deduplicated*; we take the MAX across copies to avoid double-counting.
    """
    await ctx.info(
        f"Collecting perf TS + hotspots (window={window_minutes}m, bucket={bucket}, limit_per_node={limit_per_node}, top_n={top_n}, flush={flush})"
    )

    manager = ctx.request_context.lifespan_context.get("vertica_manager")
    if not manager:
        raise RuntimeError("No database connection manager available")

    bucket = bucket.lower()
    if bucket not in {"second", "minute", "hour"}:
        raise ValueError("bucket must be one of: second, minute, hour")

    def _rows_to_dicts(cur, rows):
        cols = [d[0] for d in cur.description] if cur.description else []
        return [dict(zip(cols, r)) for r in rows]

    conn = cursor = None
    try:
        conn = manager.get_connection()
        cursor = conn.cursor()

        if flush:
            try:
                cursor.execute("SELECT FLUSH_DATA_COLLECTOR();")
                _ = cursor.fetchall()
                await ctx.info("FLUSH_DATA_COLLECTOR() executed.")
            except Exception as e:
                await ctx.info(f"FLUSH_DATA_COLLECTOR() not executed: {e}")

        # Common parameters
        window_minutes = max(1, int(window_minutes))
        top_n = max(1, int(top_n))
        ts_expr = f"DATE_TRUNC('{bucket}', end_time)"
        where = f"end_time >= CURRENT_TIMESTAMP - INTERVAL '{window_minutes} minutes'"

        # ---- CPU TS
        cpu_sql = f"""
            SELECT node_name,
                   {ts_expr} AS ts,
                   AVG(average_cpu_usage_percent) AS cpu_pct
            FROM v_monitor.cpu_usage
            WHERE {where}
            GROUP BY node_name, ts
        """
        if limit_per_node:
            cpu_sql = f"""
                SELECT node_name, ts, cpu_pct
                FROM (
                  SELECT node_name, ts, cpu_pct,
                         ROW_NUMBER() OVER (PARTITION BY node_name ORDER BY ts DESC) AS rn
                  FROM ({cpu_sql}) s
                ) t
                WHERE rn <= {int(limit_per_node)}
            """
        cpu_sql += " ORDER BY node_name, ts"

        # ---- Memory TS
        mem_sql = f"""
            SELECT node_name,
                   {ts_expr} AS ts,
                   AVG(average_memory_usage_percent) AS mem_pct
            FROM v_monitor.memory_usage
            WHERE {where}
            GROUP BY node_name, ts
        """
        if limit_per_node:
            mem_sql = f"""
                SELECT node_name, ts, mem_pct
                FROM (
                  SELECT node_name, ts, mem_pct,
                         ROW_NUMBER() OVER (PARTITION BY node_name ORDER BY ts DESC) AS rn
                  FROM ({mem_sql}) s
                ) t
                WHERE rn <= {int(limit_per_node)}
            """
        mem_sql += " ORDER BY node_name, ts"

        # ---- Network TS
        net_sql = f"""
            SELECT node_name,
                   {ts_expr} AS ts,
                   AVG(tx_kbytes_per_sec) AS tx_kbytes_per_sec,
                   AVG(rx_kbytes_per_sec) AS rx_kbytes_per_sec
            FROM v_monitor.network_usage
            WHERE {where}
            GROUP BY node_name, ts
        """
        if limit_per_node:
            net_sql = f"""
                SELECT node_name, ts, tx_kbytes_per_sec, rx_kbytes_per_sec
                FROM (
                  SELECT node_name, ts, tx_kbytes_per_sec, rx_kbytes_per_sec,
                         ROW_NUMBER() OVER (PARTITION BY node_name ORDER BY ts DESC) AS rn
                  FROM ({net_sql}) s
                ) t
                WHERE rn <= {int(limit_per_node)}
            """
        net_sql += " ORDER BY node_name, ts"

        # ---- Resource Pool snapshot (current)
        pools_sql = """
            SELECT
              node_name,
              pool_name,
              running_query_count,
              planned_concurrency,
              max_concurrency,
              memory_size_kb,
              memory_size_actual_kb,
              memory_inuse_kb,
              general_memory_borrowed_kb,
              max_memory_size_kb,
              max_query_memory_size_kb,
              queue_timeout_in_seconds,
              is_standalone
            FROM v_monitor.resource_pool_status
            ORDER BY pool_name, node_name
        """

        # ---- Top-N tables by ROS containers (sum across projections & nodes)
        top_tables_sql = f"""
            SELECT
              anchor_table_schema,
              anchor_table_name,
              SUM(ros_count) AS total_ros_containers,
              SUM(used_bytes) AS used_bytes
            FROM v_monitor.projection_storage
            GROUP BY 1,2
            ORDER BY total_ros_containers DESC
            LIMIT {top_n}
        """

        # ---- Top-N projections by ROS containers (de-dup _b0/_b1 copies)
        top_projs_sql = f"""
            WITH proj AS (
              SELECT
                projection_schema,
                projection_name,
                SUM(ros_count) AS ros_count_sum,
                SUM(used_bytes) AS used_bytes_sum
              FROM v_monitor.projection_storage
              GROUP BY 1,2
            ),
            normalized AS (
              SELECT
                projection_schema,
                REGEXP_REPLACE(projection_name, '_b[01]$', '') AS base_projection_name,
                ros_count_sum,
                used_bytes_sum
              FROM proj
            )
            SELECT
              projection_schema,
              base_projection_name,
              MAX(ros_count_sum) AS total_ros_containers,
              MAX(used_bytes_sum) AS used_bytes
            FROM normalized
            GROUP BY 1,2
            ORDER BY total_ros_containers DESC
            LIMIT {top_n}
        """

        # Execute in order
        cursor.execute(cpu_sql)
        cpu_rows = _rows_to_dicts(cursor, cursor.fetchall())

        cursor.execute(mem_sql)
        mem_rows = _rows_to_dicts(cursor, cursor.fetchall())

        cursor.execute(net_sql)
        net_rows = _rows_to_dicts(cursor, cursor.fetchall())

        cursor.execute(pools_sql)
        pools_rows = _rows_to_dicts(cursor, cursor.fetchall())

        cursor.execute(top_tables_sql)
        top_tables_rows = _rows_to_dicts(cursor, cursor.fetchall())

        cursor.execute(top_projs_sql)
        top_projs_rows = _rows_to_dicts(cursor, cursor.fetchall())

        return {
            "cpu": cpu_rows,
            "memory": mem_rows,
            "network": net_rows,
            "resource_pools": pools_rows,
            "top_tables_by_ros": top_tables_rows,
            "top_projections_by_ros": top_projs_rows,
            "meta": {
                "window_minutes": window_minutes,
                "bucket": bucket,
                "limit_per_node": limit_per_node,
                "top_n": top_n,
            },
        }

    finally:
        if cursor:
            cursor.close()
        if conn:
            manager.release_connection(conn)


@mcp.prompt()
async def sql_query_safety_guard() -> str:
    """
    🛡️ SQL Query Safety Guard - Prevents accidental large result sets and provides safe query execution workflow.
    """
    return """When the user asks to run a SQL query:
1) Always call the tool `run_query_safely` with the user's SQL and row_threshold=1000.
2) If the tool returns `requires_confirmation: true`, ASK the user: 
   "⚠️ This query will return a large result set (possibly millions of rows). Proceed with paged results?"
   Then:
     - If user says yes, call `run_query_safely` again with `proceed=true`, `mode="page"`.
     - If user asks for file export instead, propose paging or suggest they use a smaller LIMIT first.
3) Never request more than one page at a time unless the user explicitly asks.
4) For DDL/DML operations, remind users about the configured permissions."""


@mcp.prompt()
async def vertica_query_performance_analyzer() -> str:
    """
    🚀 Vertica Performance Analyzer - Deep-dive query performance analysis with actionable optimization recommendations.

    Analyzes the given query execution plans, identifies bottlenecks, and provides concrete DDL suggestions
    for optimal Vertica projections, join strategies, and ROS container health.
    """
    return """🔍 VERTICA PERFORMANCE ANALYSIS

ANALYSIS WORKFLOW:
1) **PROFILE EXECUTION**: Call `profile_query` to get actual runtime and execution plan
2) **PARSE OPERATORS**: Identify all operators (Scan, Hash Join, Merge Join, GroupBy, Sort, Exchange)
3) **COST ANALYSIS**: Find the 3-5 highest-cost operations with row estimates
4) **JOIN OPTIMIZATION**: For each Hash Join → propose Merge Join strategy
5) **AGGREGATION TUNING**: For each Hash GroupBy → propose Pipelined GroupBy
6) **PROJECTION HEALTH**: Audit ROS container counts for projections used
7) **CONCRETE RECOMMENDATIONS**: Specific DDL for optimal projections

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


@mcp.prompt()
async def vertica_database_health_dashboard() -> str:
    """
    📊 Vertica Health Dashboard - Comprehensive database status and usage analytics.

    Generates visual dashboards showing database version, storage usage, performance trends,
    and capacity utilization with charts and key metrics.
    """
    return """📊 VERTICA DATABASE HEALTH DASHBOARD

WORKFLOW:
1. Call `database_status` tool to get current metrics
2. Generate comprehensive dashboard with visualizations

DASHBOARD SECTIONS:

🔹 **Database Overview**
- Vertica version and build information
- Current database name and connection status

🔹 **Storage Analytics** (with charts)
- Total database size (GB, 2 decimal places)
- License utilization vs. capacity
- Growth trends over time
- Storage efficiency metrics

🔹 **Usage Statistics** (with visualizations)
- Database capacity utilization percentage
- Historical usage patterns
- Peak usage periods
- Audit timestamp analysis

🔹 **Performance Indicators**
- Query throughput metrics
- Resource utilization highlights
- System health indicators

VISUALIZATION REQUIREMENTS:
📈 Use clear, labeled charts for all metrics
📈 Format all sizes as GB with 2 decimal places (e.g., 15.73 GB)
📈 Include proper titles, legends, and units
📈 Use distinct colors and styles for clarity
📈 Focus on actionable insights

OUTPUT FORMAT:
- Executive summary with key findings
- Visual charts for trends and comparisons
- Tabular data for detailed metrics
- Recommendations for optimization"""


@mcp.prompt()
async def vertica_database_system_monitor() -> str:
    """
    ⚡ Vertica System Monitor - Real-time performance monitoring with resource analysis.

    Fast dashboard showing CPU, memory, network usage, resource pools, and top consumers
    with immediate insights and alerts for system optimization.
    """
    return """⚡ VERTICA SYSTEM PERFORMANCE MONITOR

GOAL: Quick system health check with actionable insights

WORKFLOW:
1. Call `analyze_system_performance` for real-time metrics
2. Generate fast, focused dashboard

DASHBOARD COMPONENTS:

📊 **Resource Metrics** (Time-series plots)
- CPU usage by node (line chart: time vs. cpu_pct)
- Memory utilization by node (line chart: time vs. mem_pct) 
- Network throughput (line chart: TX/RX kbytes/sec by node)

📋 **Resource Analysis** (Tables)
- Active resource pools: name, node, running queries, memory usage
- Top storage consumers: tables with ROS container counts
- Projection hotspots: high ROS count projections (flag >5000)

🚨 **Immediate Alerts**
- Nodes with CPU/Memory >85% average or >95% peak
- Resource pools exceeding concurrency limits  
- Projections requiring mergeout (>5000 ROS containers)

OUTPUT STRUCTURE:
# System Performance Dashboard
## Resource Charts (CPU/Memory/Network)
## Resource Analysis Tables  
## Critical Actions Required

FOCUS: Speed and clarity - highlight problems requiring immediate attention"""
