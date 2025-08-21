# Copyright 2025 Abdelhak Zabour
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CLI for MCP Vertica Server
This module provides a command-line interface for the MCP Vertica Server,
allowing users to configure and run the server with various options.
It supports loading environment variables from a .env file, setting up logging,
and configuring database connection parameters.
"""

import asyncio
import logging
import os

import click
from dotenv import load_dotenv

from .connection import (VERTICA_CONNECTION_LIMIT, VERTICA_DATABASE,
                         VERTICA_HOST, VERTICA_PASSWORD, VERTICA_PORT,
                         VERTICA_SSL, VERTICA_SSL_REJECT_UNAUTHORIZED,
                         VERTICA_USER)
from .server import mcp, run_http, run_sse
from .utils import setup_logger

load_dotenv()


def main(
    verbose: int,
    env_file: str | None,
    transport: str,
    port: int,
    host: str | None,  # ← Now can be None
    bind_host: str,  # ← Separate binding host
    db_port: int | None,
    database: str | None,
    user: str | None,
    password: str | None,
    connection_limit: int | None,
    ssl: bool | None,
    ssl_reject_unauthorized: bool | None,
    http_path: str,
    http_json: bool,
    http_stateless: bool,
) -> None:
    """MCP Vertica Server - Vertica functionality for MCP"""
    setup_logger(verbose)

    # Load .env first
    if env_file:
        logging.debug("Loading environment from file: %s", env_file)
        load_dotenv(env_file)
    else:
        logging.debug("Loading environment from default .env file")
        load_dotenv()

    # Set environment variables from CLI args (only if provided)
    if host:  # ← Only override if explicitly provided
        os.environ[VERTICA_HOST] = host
    if db_port:
        os.environ[VERTICA_PORT] = str(db_port)
    if database:
        os.environ[VERTICA_DATABASE] = database
    if user:
        os.environ[VERTICA_USER] = user
    if password:
        os.environ[VERTICA_PASSWORD] = password
    if connection_limit:
        os.environ[VERTICA_CONNECTION_LIMIT] = str(connection_limit)
    if ssl is not None:
        os.environ[VERTICA_SSL] = str(ssl).lower()
    if ssl_reject_unauthorized is not None:
        os.environ[VERTICA_SSL_REJECT_UNAUTHORIZED] = str(
            ssl_reject_unauthorized
        ).lower()

    if transport.lower() == "sse":
        logging.info(f"Launching SSE transport on {bind_host}:{port}")
        asyncio.run(run_sse(host=bind_host, port=port))
    elif transport.lower() in ("http", "streamable-http"):
        logging.info(
            f"Launching Streamable HTTP on {bind_host}:{port}{http_path} "
            f"(json_response={http_json}, stateless={http_stateless})"
        )
        asyncio.run(
            run_http(
                host=bind_host,
                port=port,
                path=http_path,
                json_response=http_json,
                stateless_http=http_stateless,
            )
        )
    else:
        mcp.run()  # stdio


@click.command()
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity (can be used multiple times, e.g., -v, -vv, -vvv)",
)
@click.option(
    "--env-file", type=click.Path(exists=True, dir_okay=False), help="Path to .env file"
)
@click.option(
    "--transport",
    type=click.Choice(
        ["stdio", "sse", "http", "streamable-http"], case_sensitive=False
    ),
    default="stdio",
    help="Transport type (stdio, sse, http/streamable-http).",
)
@click.option(
    "--port",
    default=8000,
    help="Port to listen on for SSE/HTTP transport",
)
@click.option(
    "--host",
    default=None,  # ← Changed from "localhost" to None
    help="Host to bind to for SSE/HTTP transport, or Vertica host for database",
)
@click.option(
    "--bind-host",  # ← Add separate option for binding
    default="localhost",
    help="Host to bind SSE/HTTP server to (default: localhost)",
)
@click.option(
    "--db-port",
    type=int,
    help="Vertica port",
)
@click.option(
    "--database",
    help="Vertica database name",
)
@click.option(
    "--user",
    help="Vertica username",
)
@click.option(
    "--password",
    help="Vertica password",
)
@click.option(
    "--connection-limit",
    type=int,
    default=10,
    help="Maximum number of connections in the pool",
)
@click.option(
    "--ssl",
    is_flag=True,
    default=False,
    help="Enable SSL for database connection",
)
@click.option(
    "--ssl-reject-unauthorized",
    is_flag=True,
    default=True,
    help="Reject unauthorized SSL certificates",
)
@click.option(
    "--http-path",
    default="/mcp",
    show_default=True,
    help="Endpoint path for Streamable HTTP.",
)
@click.option(
    "--http-json/--no-http-json",
    default=False,
    show_default=True,
    help="Return batch JSON responses when possible (otherwise stream).",
)
@click.option(
    "--http-stateless/--http-stateful",
    default=True,
    show_default=True,
    help="Use stateless Streamable HTTP sessions (recommended for remote clients).",
)
def cli(**kwargs):
    """Command-line interface for MCP Vertica Server
    This function sets up the command-line interface using Click and calls the main function.
    """
    main(**kwargs)
