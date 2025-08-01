"""Server module for Vertica MCP. Provides API endpoints and database utilities."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Any, List
import logging
import re
import csv
import io
from mcp.server.fastmcp import FastMCP, Context
from starlette.applications import Starlette
from starlette.routing import Mount
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


@asynccontextmanager
async def server_lifespan(_server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """Server lifespan context manager that handles initialization and cleanup.

    Args:
        server: FastMCP server instance

    Yields:
        Dictionary containing the Vertica connection manager
    """
    manager = None
    try:
        # Initialize Vertica connection manager
        manager = VerticaConnectionManager()
        config = VerticaConfig.from_env()
        manager.initialize_default(config)
        logger.info("Vertica connection manager initialized")
        yield {"vertica_manager": manager}
    except Exception as e:
        logger.error("Failed to initialize server: %s", str(e))
        raise
    finally:
        # Cleanup resources
        if manager:
            try:
                manager.close_all()
                logger.info("Vertica connection manager closed")
            except Exception as e:
                logger.error("Error during cleanup: %s", str(e))


# Create FastMCP instance with SSE support
mcp = FastMCP(
    MCP_SERVER_NAME,
    dependencies=DEPENDENCIES,
    lifespan=server_lifespan,
)


async def run_sse(port: int = 8000) -> None:
    """Run the MCP server with SSE transport.

    Args:
        port: Port to listen on for SSE transport
    """
    starlette_app = Starlette(routes=[Mount("/", app=mcp.sse_app())])
    config = uvicorn.Config(starlette_app, host="0.0.0.0", port=port)  # noqa: S104
    app = uvicorn.Server(config)
    await app.serve()


@mcp.tool()
async def execute_query(ctx: Context, query: str) -> dict:
    """Execute a SQL query and return the results.

    Args:
        ctx (Context): FastMCP context for progress reporting and logging.
        query (str): SQL query to execute.

    Returns:
        dict: Dictionary containing results and row count.
        If an error occurs, returns an error message.
    """
    await ctx.info(f"Executing query: {query}")
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
        results = cursor.fetchall()
        await ctx.info(f"Query executed successfully, returned {len(results)} rows")
        return {"result": results, "row_count": len(results)}
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
async def stream_query(ctx: Context, query: str, batch_size: int = 1000) -> dict:
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
async def copy_data(
    ctx: Context, schema: str, table: str, data: List[List[Any]]
) -> dict:
    """Copy data into a Vertica table using the COPY command.

    Args:
        ctx (Context): FastMCP context for progress reporting and logging.
        schema (str): Vertica schema to execute the copy against.
        table (str): Target table name.
        data (List[List[Any]]): List of rows to insert.

    Returns:
        dict: Dictionary containing success message, number of rows copied, and table name.
        If an error occurs, returns an error message.
    """
    await ctx.info(f"Copying {len(data)} rows to table: {table}")
    manager = ctx.request_context.lifespan_context.get("vertica_manager")
    if not manager:
        await ctx.error("No database connection manager available")
        raise RuntimeError("No database connection manager available")
    if not manager.is_operation_allowed(schema, OperationType.INSERT):
        error_msg = f"INSERT operation not allowed for database {schema}"
        await ctx.error(error_msg)
        raise RuntimeError(error_msg)
    conn = None
    cursor = None
    try:
        conn = manager.get_connection()
        cursor = conn.cursor()
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
        writer.writerows(data)
        output.seek(0)
        copy_query = f"COPY {table} FROM STDIN DELIMITER ',' ENCLOSED BY '\"'"
        cursor.copy(copy_query, output.getvalue())
        conn.commit()
        success_msg = f"Successfully copied {len(data)} rows to {table}"
        await ctx.info(success_msg)
        return {"result": success_msg, "rows_copied": len(data), "table": table}
    except Exception as e:
        error_msg = f"Error copying data: {str(e)}"
        await ctx.error(error_msg)
        raise RuntimeError(error_msg) from e
    finally:
        if cursor:
            cursor.close()
        if conn:
            manager.release_connection(conn)


@mcp.tool()
async def get_table_structure(
    ctx: Context, table_name: str, schema: str = "public"
) -> dict:
    """Get the structure of a Vertica table, including columns, data types, and constraints.

    Args:
        ctx (Context): FastMCP context for progress reporting and logging.
        table_name (str): Name of the table to inspect.
        schema (str, optional): Schema name (default: "public").

    Returns:
        dict: Dictionary containing table structure information.
        If an error occurs, returns an error message.
    """
    await ctx.info(f"Getting structure for table: {schema}.{table_name}")
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
        cursor.execute(query, (schema, table_name))
        columns = cursor.fetchall()
        if not columns:
            raise RuntimeError(f"No table found: {schema}.{table_name}")
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
            (schema, table_name),
        )
        constraints = cursor.fetchall()
        result = f"Table Structure for {schema}.{table_name}:\n\n"
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
            "schema": schema,
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
async def list_projections(
    ctx: Context, table_name: str, schema: str = "public"
) -> dict:
    """List all projections for a specific Vertica table.

    Args:
        ctx (Context): FastMCP context for progress reporting and logging.
        table_name (str): Name of the table to inspect.
        schema (str, optional): Schema name (default: "public").

    Returns:
        dict: Dictionary containing projections information.
        If an error occurs, returns an error message.
    """
    await ctx.info(f"Listing projections for table: {schema}.{table_name}")
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
        cursor.execute(query, (schema, table_name))
        projections = cursor.fetchall()
        if not projections:
            raise RuntimeError(f"No projections found for table: {schema}.{table_name}")
        result = f"Projections for {schema}.{table_name}:\n\n"
        for proj in projections:
            result += f"- {proj[0]} (Super Projection: {proj[1]}) [Table: {proj[2]}] (Creation Type: {proj[3]})\n"
        return {
            "result": result,
            "table_name": table_name,
            "schema": schema,
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
async def list_views(ctx: Context, schema: str = "public") -> dict:
    """List all views in a specific Vertica schema.

    Args:
        ctx (Context): FastMCP context for progress reporting and logging.
        schema (str, optional): Schema name (default: "public").

    Returns:
        dict: Dictionary containing views information.
        If an error occurs, returns an error message.
    """
    await ctx.info(f"Listing views in schema: {schema}")
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
        cursor.execute(query, (schema,))
        views = cursor.fetchall()
        if not views:
            raise RuntimeError(f"No views found in schema: {schema}")
        result = f"Views in schema {schema}:\n\n"
        for view in views:
            result += f"View: {view[0]}\n"
            result += f"Definition:\n{view[1]}\n\n"
        return {"result": result, "schema": schema, "view_count": len(views)}
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
async def list_tables(ctx: Context, schema: str = "public") -> dict:
    """List all tables in a specific Vertica schema.

    Args:
        ctx (Context): FastMCP context for progress reporting and logging.
        schema (str, optional): Schema name (default: "public").

    Returns:
        dict: Dictionary containing tables information.
        If an error occurs, returns an error message.
    """
    await ctx.info(f"Listing tables in schema: {schema}")
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
        cursor.execute(query, (schema,))
        tables = cursor.fetchall()
        if not tables:
            raise RuntimeError(f"No tables found in schema: {schema}")
        result = f"Tables in schema {schema}:\n\n"
        for table in tables:
            result += f"{table[0]}\n"
        return {"result": result, "schema": schema, "table_count": len(tables)}
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
async def list_schemas(ctx: Context) -> dict:
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
    FROM schemata
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
async def explain_query(ctx: Context, query: str) -> dict:
    """Explain the execution plan of a SQL query and return the query plan.

    Args:
        ctx (Context): FastMCP context for progress reporting and logging.
        query (str): SQL query to explain.

    Returns:
        dict: Dictionary containing query plan.
        If an error occurs, returns an error message.
    """
    await ctx.info(f"Explaining query: {query}")
    manager = ctx.request_context.lifespan_context.get("vertica_manager")
    if not manager:
        await ctx.error("No database connection manager available")
        raise RuntimeError("No database connection manager available")
    conn = None
    cursor = None
    try:
        conn = manager.get_connection()
        cursor = conn.cursor()
        cursor.execute(f"EXPLAIN {query}")
        plan_rows = cursor.fetchall()
        if not plan_rows:
            raise RuntimeError("Could not retrieve query plan.")
        # Convert the plan rows to a readable string, keeping only the Access Path section
        plan_lines = []
        in_access_path = False

        for row in plan_rows:
            line = row[0]

            # Start capturing when we find "Access Path:"
            if "Access Path:" in line:
                in_access_path = True
                plan_lines.append(line)
                continue

            # Stop capturing when we hit the end markers
            if in_access_path and (
                "------------------------------" in line
                or "-----------------------------------------------" in line
            ):
                break

            # Add lines while we're in the Access Path section
            if in_access_path:
                plan_lines.append(line)

        plan_text = "\n".join(plan_lines)

        return {"query": query, "plan": plan_text, "plan_lines": len(plan_lines)}
    except Exception as e:
        error_msg = f"Error explaining query: {str(e)}"
        await ctx.error(error_msg)
        raise RuntimeError(error_msg) from e
    finally:
        if cursor:
            cursor.close()
        if conn:
            manager.release_connection(conn)


@mcp.tool()
async def profile_query(ctx: Context, query: str) -> dict:
    """Profile a SQL query and return the query plan.

    Args:
        ctx (Context): FastMCP context for progress reporting and logging.
        query (str): SQL query to profile.

    Returns:
        dict: Dictionary containing query plan and execution duration.
        If an error occurs, returns an error message.
    """
    await ctx.info(f"Profiling query: {query}")
    manager = ctx.request_context.lifespan_context.get("vertica_manager")
    if not manager:
        await ctx.error("No database connection manager available")
        raise RuntimeError("No database connection manager available")
    conn = None
    cursor = None
    try:
        conn = manager.get_connection()
        cursor = conn.cursor()
        cursor.execute(f"PROFILE {query}")
        cursor.execute(
            "SELECT transaction_id, statement_id FROM v_monitor.current_session"
        )
        ids = cursor.fetchall()
        if not ids:
            raise RuntimeError("Could not retrieve transaction/statement IDs.")
        trxid, stmtid = ids
        cursor.execute(
            "SELECT query_duration_us FROM v_monitor.query_profiles WHERE transaction_id = %s AND statement_id = %s",
            (trxid, stmtid),
        )
        duration = cursor.fetchone()
        duration_str = (
            f"Execution Duration (us): {duration[0]}"
            if duration
            else "Duration not found."
        )
        cursor.execute(
            """SELECT path_line 
                FROM v_internal.dc_explain_plans 
                WHERE transaction_id = %s 
                AND statement_id = %s 
                ORDER BY path_id, 
                        path_line_index""",
            (trxid, stmtid),
        )
        plan_lines = cursor.fetchall()
        plan_str = (
            "\n".join(line[0] for line in plan_lines)
            if plan_lines
            else "Query plan not found."
        )
        result = f"{duration_str}\n\nQuery Plan:\n{plan_str}"
        await ctx.info("Profile completed.")
        return {
            "result": result,
            "query": query,
            "duration_us": duration[0] if duration else None,
            "plan_line_count": len(plan_lines) if plan_lines else 0,
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

@mcp.prompt()
async def analyze_query_performance(query: str) -> str:
    """
    Analyze the performance of a query and suggest improvements.

    Args:
        query (str): The SQL query to analyze.

    Returns:
    """

    logger.info("Providing analyze_query_performance prompt for query: %s", query)
    return f"""Query: {query}

        Please analyze this query for performance issues:

        1. First, use the explain_query tool to get the execution plan
        2. Look for projection scans instead 
        3. Check if the joins, group by are efficient (if they are merge joins and pipelined group by)
        4. Check the cost of each operation
        5. Identify if the query can be optimized with better projection
        6. Suggest concrete improvements to make the query more efficient
        """


@mcp.prompt()
async def write_query_for_task(task: str) -> str:
    """
    Write an SQL query for a given task.

    Args:
        task (str): The task to write a query for.

    Returns:
    """
    logger.info("Providing write_query_for_task prompt for task: %s", task)
    return f"""Task: {task}

        Please write an SQL query that accomplishes this task efficiently.

        Some guidelines:
        1. Use appropriate JOINs (INNER, LEFT, RIGHT) based on the data relationships
        2. Filter data in the WHERE clause to minimize data processing
        3. Consider using projections for better performance in Vertica
        4. Use appropriate aggregation functions when needed
        5. Format the query with clear indentation for readability

        If you need to see the database schema first, you can use the list_tables, list_schemas, and get_table_structure tools to explore the database structure.
        """

@mcp.prompt()
async def database_status_prompt() -> str:
    """
    Prompt for the database status.
    """
    return """ When a user asks about the database status, you can use the database_status tool to get the status of the database.
    Use these status + statistics to create charts and a plots for a report of the database status.
    The report should be a markdown file with the following sections and visualizations:
    - Database Version
    - Database Size 
    - Database Usage Statistics (create a chart to visualize the usage statistics)
    - Database Usage Percentage (create a chart to visualize the usage percentage)
    - Database Usage Percentage (create a chart to visualize the usage percentage)
    """


# @mcp.resource("schema://{schema_name}")
# def get_schema_info(schema_name: str = "public") -> str:
#     """Get comprehensive schema information for a Vertica schema"""
#     logger.info("Getting schema information for schema: %s", schema_name)

#     manager = None
#     conn = None
#     cursor = None
#     try:
#         manager = VerticaConnectionManager()
#         config = VerticaConfig.from_env()
#         manager.initialize_default(config)

#         conn = manager.get_connection()
#         cursor = conn.cursor()

#         schema_info = []
#         schema_info.append(f"=== SCHEMA: {schema_name} ===\n")

#         # Get tables
#         cursor.execute(
#             """
#             SELECT table_name 
#             FROM v_catalog.tables 
#             WHERE table_schema = %s 
#             ORDER BY table_name
#         """,
#             (schema_name,),
#         )
#         tables = cursor.fetchall()

#         if tables:
#             schema_info.append(f"TABLES ({len(tables)}):")
#             for table in tables:
#                 table_name = table[0]
#                 schema_info.append(f"\n--- TABLE: {table_name} ---")

#                 # Get table structure
#                 cursor.execute(
#                     """
#                     SELECT column_name, data_type, character_maximum_length, 
#                            numeric_precision, numeric_scale, is_nullable, column_default
#                     FROM v_catalog.columns 
#                     WHERE table_schema = %s AND table_name = %s 
#                     ORDER BY ordinal_position
#                 """,
#                     (schema_name, table_name),
#                 )
#                 columns = cursor.fetchall()

#                 schema_info.append("Columns:")
#                 for col in columns:
#                     col_info = f"  {col[0]}: {col[1]}"
#                     if col[2]:  # character_maximum_length
#                         col_info += f"({col[2]})"
#                     elif col[3]:  # numeric_precision
#                         col_info += f"({col[3]},{col[4]})"
#                     col_info += f" {'NULL' if col[5] == 'YES' else 'NOT NULL'}"
#                     if col[6]:  # column_default
#                         col_info += f" DEFAULT {col[6]}"
#                     schema_info.append(col_info)

#                 # Get projections
#                 cursor.execute(
#                     """
#                     SELECT projection_name, is_super_projection, create_type
#                     FROM v_catalog.projections 
#                     WHERE projection_schema = %s AND anchor_table_name = %s
#                     ORDER BY projection_name
#                 """,
#                     (schema_name, table_name),
#                 )
#                 projections = cursor.fetchall()

#                 if projections:
#                     schema_info.append("Projections:")
#                     for proj in projections:
#                         schema_info.append(
#                             f"  {proj[0]} (Super: {proj[1]}, Type: {proj[2]})"
#                         )

#         # Get views
#         cursor.execute(
#             """
#             SELECT table_name, view_definition 
#             FROM v_catalog.views 
#             WHERE table_schema = %s 
#             ORDER BY table_name
#         """,
#             (schema_name,),
#         )
#         views = cursor.fetchall()

#         if views:
#             schema_info.append(f"\nVIEWS ({len(views)}):")
#             for view in views:
#                 schema_info.append(f"\n--- VIEW: {view[0]} ---")
#                 schema_info.append(f"Definition: {view[1]}")

#         return "\n".join(schema_info)

#     except Exception as e:
#         error_msg = f"Error getting schema information: {str(e)}"
#         logger.error(error_msg)
#         return f"Error: {error_msg}"
#     finally:
#         if cursor:
#             cursor.close()
#         if conn:
#             manager.release_connection(conn)
#         if manager:
#             manager.close_all()
