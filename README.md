# Vertica MCP Server

<div align="center">

![Vertica MCP Banner](https://img.shields.io/badge/Vertica_MCP-Enterprise_Analytics-00A0E3?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTggMTZDMTIuNDE4MyAxNiAxNiAxMi40MTgzIDE2IDhDMTYgMy41ODE3MiAxMi40MTgzIDAgOCAwQzMuNTgxNzIgMCAwIDMuNTgxNzIgMCA4QzAgMTIuNDE4MyAzLjU4MTcyIDE2IDggMTZ6IiBmaWxsPSIjMDBBMEUzIi8+Cjwvc3ZnPg==)

**Transform your Vertica Analytics Database into an AI-powered intelligence layer**

[![PyPI version](https://badge.fury.io/py/vertica-mcp.svg)](https://badge.fury.io/py/vertica-mcp)
<!-- [![MCP Version](https://img.shields.io/badge/MCP-2025--08--20-blue.svg)](https://spec.modelcontextprotocol.io/) -->
[![Python Version](https://img.shields.io/badge/Python-3.11+-green.svg)](https://www.python.org/)
[![Vertica Version](https://img.shields.io/badge/Vertica-24.x+-orange.svg)](https://www.vertica.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker)](https://hub.docker.com/)
<!-- [![CI/CD](https://img.shields.io/github/actions/workflow/status/zaboura/vertica-mcp/ci.yml?label=CI/CD)](https://github.com/zaboura/vertica-mcp/actions) -->
<!-- [![Python](https://img.shields.io/pypi/pyversions/vertica-mcp.svg)](https://pypi.org/project/vertica-mcp/) -->
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://pepy.tech/badge/vertica-mcp)](https://pepy.tech/project/vertica-mcp)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Quick Start](#quick-start) • [Documentation](#documentation) • [Features](#features) • [Contributing](#contributing) • [Community](#community)

</div>

---

## Why Vertica MCP?

The **Vertica MCP Server** is a production-ready implementation of the [Model Context Protocol](https://modelcontextprotocol.io/) that transforms your Vertica Analytics Database into an intelligent, AI-accessible data platform. Built with enterprise security and performance in mind, it enables AI assistants like Claude, ChatGPT, and Cursor to directly query, analyze, and optimize your Vertica databases through natural language.

### What is MCP?

The Model Context Protocol (MCP) is an open standard developed by Anthropic that provides a universal way for AI assistants to connect with external tools and data sources. Think of it as "USB-C for AI" - a standardized interface that allows any MCP-compatible AI to interact with your systems without custom integrations.

### Key Benefits

- **Universal AI Connectivity**: Connect any MCP-compatible AI to your Vertica database without custom integrations
- **Enterprise Security**: Fine-grained permissions at schema and operation levels with SSL/TLS support
- **High Performance**: Connection pooling, query streaming, and automatic pagination for handling massive datasets
- **AI-Optimized**: Built-in prompts and tools specifically designed for database analysis and optimization
- **Multiple Transports**: Support for STDIO, HTTP, and SSE to fit any deployment scenario
- **Production Ready**: Battle-tested with comprehensive error handling, logging, and monitoring

---

## Prerequisites

- **Python** 3.11 or higher
- **Vertica Database** (accessible instance)
- **uv** (Python package manager) - [Installation guide](https://github.com/astral-sh/uv)
- **Docker** (optional, for containerized deployment)
- **Claude Desktop** or another MCP-compatible client

---

## Quick Start

### Method 1: Local Installation (Development Environment)

This method is recommended when you want to modify the code or work with the development version.

```bash
# 1. Clone the repository
git clone https://github.com/zaboura/vertica-mcp.git
cd vertica-mcp

# 2. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Setup environment and install dependencies
uv sync
source .venv/bin/activate

# 4. Install in development mode
uv pip install -e .

# 5. Configure database connection
cp .env.example .env
# Edit .env with your Vertica credentials

# 6. Run the server
vertica-mcp --transport http --port 8000 --bind-host 0.0.0.0  # HTTP for remote access
```

### Method 2: PyPI Installation (Production Environment)

This method is recommended for production deployments and when you want to use the stable release.

```bash
# 1. Install from PyPI
pip install vertica-mcp

# 2. Initialize configuration
vertica-mcp --init

# 3. Edit configuration with your credentials
nano .env  # or use your preferred editor
# Update VERTICA_HOST, VERTICA_USER, VERTICA_PASSWORD, etc.

# 4. Test the installation
vertica-mcp --transport http --port 8000  # For HTTP access
```

#### Configuration File

After running `vertica-mcp --init`, edit the generated `.env` file with your specific settings:

```bash
# Required Database Connection
VERTICA_HOST=your_vertica_host
VERTICA_PORT=5433
VERTICA_DATABASE=your_database
VERTICA_USER=your_username
VERTICA_PASSWORD=your_password

# Connection Pool Configuration
VERTICA_CONNECTION_LIMIT=10
VERTICA_LAZY_INIT=1

# SSL Configuration (optional but recommended for production)
VERTICA_SSL=false
VERTICA_SSL_REJECT_UNAUTHORIZED=true

# Performance and Resource Management
VERTICA_QUERY_TIMEOUT=600  # Query timeout in seconds
VERTICA_MAX_RETRIES=3      # Max retry attempts
VERTICA_RETRY_DELAY=0.1    # Base delay for retries
VERTICA_CACHE_TTL=300      # Cache TTL in seconds
VERTICA_MAX_RESULT_MB=100  # Max result size in MB
VERTICA_RATE_LIMIT=60      # Requests per minute
VERTICA_HEALTH_CHECK_INTERVAL=60  # Health check interval

# Security Permissions (defaults to read-only for safety)
ALLOW_INSERT_OPERATION=false
ALLOW_UPDATE_OPERATION=false
ALLOW_DELETE_OPERATION=false
ALLOW_DDL_OPERATION=false

# Schema-specific Permissions (optional for granular control)
SCHEMA_INSERT_PERMISSIONS=staging:true,production:false
SCHEMA_UPDATE_PERMISSIONS=staging:true,production:false
SCHEMA_DELETE_PERMISSIONS=staging:false,production:false
SCHEMA_DDL_PERMISSIONS=staging:false,production:false
```

#### Testing with MCP Inspector

The MCP Inspector is a valuable tool for testing and debugging your server configuration:

```bash
# 1. Install MCP Inspector
npm install -g @modelcontextprotocol/inspector

# 2. Start your server in one terminal
vertica-mcp --transport http --port 8000

# 3. Test with inspector in another terminal
mcp-inspector http://localhost:8000/mcp
```

The inspector will open at `http://localhost:6274` where you can:

- View available database tools and their schemas
- Test tool execution interactively with real data
- Validate MCP protocol compliance
- Debug connection issues and error responses

For STDIO testing (not recommended due to command complexity), use HTTP transport which provides identical functionality validation with better debugging capabilities.

### Method 3: Docker Deployment

Docker deployment is ideal for containerized environments and consistent deployments across different systems.

#### Build Docker Image

```bash
# Build from Dockerfile
docker build -t vertica-mcp:latest .

# Or use docker-compose for easier management
docker-compose build
```

#### Run with Docker Compose

```bash
# STDIO transport (for direct MCP client connection)
docker-compose up mcp-stdio

# HTTP transport (for web-based access)
docker-compose up mcp-http

# SSE transport (for real-time streaming)
docker-compose up mcp-sse
```

#### Manual Docker Run

```bash
# HTTP transport with port mapping
docker run -d \
  --name vertica-mcp-http \
  -p 8000:8000 \
  --env-file .env \
  -e TRANSPORT=http \
  -e BIND=0.0.0.0 \
  -e PORT=8000 \
  -e HTTP_PATH=/mcp \
  vertica-mcp:latest

# STDIO transport (direct MCP client connection)
docker run -i --rm \
  --name vertica-mcp-stdio \
  --env-file .env \
  vertica-mcp:latest
```

---

## Features

### Core Tools

<table>
<tr>
<td width="50%">

#### Query Execution
- `run_query_safely` - Smart query execution with large result detection and automatic warnings
- `execute_query_paginated` - Efficient pagination for large datasets with configurable page sizes
- `execute_query_stream` - Real-time streaming for massive results with memory-efficient processing

</td>
<td width="50%">

#### Schema Management
- `get_database_schemas` - Explore database organization and available schemas
- `get_schema_tables` - List tables with metadata including row counts and storage information
- `get_table_structure` - Detailed column information, data types, constraints, and indexes
- `get_table_projections` - Vertica-specific projection analysis and optimization recommendations
- `get_schema_views` - List all views in a schema with their definitions

</td>
</tr>
<tr>
<td width="50%">

#### Performance Analysis
- `profile_query` - Query execution plans, performance metrics, and optimization suggestions
- `analyze_system_performance` - Real-time resource monitoring and system health metrics
- `database_status` - Comprehensive health metrics including storage usage and connection statistics

</td>
<td width="50%">

#### AI-Powered Prompts
- **SQL Safety Guard** - Prevents accidental large queries and suggests safer alternatives
- **Performance Analyzer** - Deep query optimization analysis with specific recommendations
- **SQL Assistant** - Intelligent query generation based on natural language descriptions
- **Health Dashboard** - Visual database insights with key performance indicators
- **System Monitor** - Real-time performance tracking with alerting capabilities

</td>
</tr>
</table>

### Security Features

- **Multi-Level Permissions**: Global and schema-specific access controls with fine-grained operation restrictions
- **SSL/TLS Encryption**: Secure database connections with certificate validation
- **Connection Pooling**: Efficient resource management with configurable limits and automatic cleanup
- **Read-Only Mode**: Default safe configuration for production environments
- **OAuth Support**: Enterprise authentication integration for remote deployments

---

## Documentation

### Configuration

<details>
<summary><b>Environment Variables (.env)</b></summary>

```bash
# Database Connection (Required)
VERTICA_HOST=your_vertica_host
VERTICA_PORT=5433
VERTICA_DATABASE=your_database
VERTICA_USER=your_username
VERTICA_PASSWORD=your_password

# Connection Pool Configuration (Optional)
VERTICA_CONNECTION_LIMIT=10
VERTICA_LAZY_INIT=1  # Delay connection until first use

# SSL Configuration (Optional but recommended for production)
VERTICA_SSL=false
VERTICA_SSL_REJECT_UNAUTHORIZED=true

# Security Permissions (Optional - defaults to read-only for safety)
ALLOW_INSERT_OPERATION=false
ALLOW_UPDATE_OPERATION=false
ALLOW_DELETE_OPERATION=false
ALLOW_DDL_OPERATION=false

# Schema-specific Permissions (Optional for granular control)
SCHEMA_INSERT_PERMISSIONS=staging:true,production:false
SCHEMA_UPDATE_PERMISSIONS=staging:true,production:false
SCHEMA_DELETE_PERMISSIONS=staging:false,production:false
SCHEMA_DDL_PERMISSIONS=staging:false,production:false
```

</details>

### Client Integration

<details>
<summary><b>Claude Desktop Configuration</b></summary>

### Claude Desktop (STDIO) — Installed Package in a Virtual Environment [Recommended]

This is the best practice approach using a dedicated Python virtual environment and the installed package for maximum stability and isolation.

To install the package and create/configure your `.env`, follow **[Method 2: PyPI Installation](#method-2-pypi-installation-production-environment)** above.

1. **Locate the Claude configuration file**
   - **Windows**: `%APPDATA%/Claude/claude_desktop_config.json`
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

2. **Configure Claude to use your virtual environment executable and configuration**
   - Replace the `command` path with your virtual environment path
   - Keep `--transport stdio` and `--env-file` with absolute path for reliability

**Windows Configuration Example**
```json
{
  "mcpServers": {
    "vertica-mcp-stdio": {
      "command": "C:\\path-to-venv\\Scripts\\vertica-mcp.exe",
      "args": ["--transport", "stdio", "--env-file", "C:\\path\\to\\.env"]
    }
  }
}
```

**macOS/Linux Configuration Example**
```json
{
  "mcpServers": {
    "vertica-mcp-stdio": {
      "command": "/Users/you/.venvs/vertica-mcp/bin/vertica-mcp",
      "args": ["--transport", "stdio", "--env-file", "/absolute/path/to/.env"]
    }
  }
}
```

**Verification Test (outside Claude)**

Test your configuration before integrating with Claude:

```bash
# Windows
C:\venv\vertica-mcp\Scripts\vertica-mcp.exe --transport stdio --env-file C:\path\to\.env -vvv

# macOS/Linux
~/.venv/vertica-mcp/bin/vertica-mcp --transport stdio --env-file /absolute/path/to/.env -vvv
```

**Important Configuration Notes**
- `--transport stdio` runs the server over STDIO (no network ports required)
- `--env-file` ensures your credentials load correctly even if Claude's working directory differs
- Use absolute paths to avoid path resolution issues

3. **Development Alternative: From Source (uv)**

This option is suitable for development when you want to work with the source code directly:

```json
{
  "mcpServers": {
    "vertica-mcp-stdio": {
      "command": "uv",
      "args": ["run", "vertica-mcp"],
      "cwd": "/path/to/vertica-mcp",
      "env": {
        "VERTICA_HOST": "your_host",
        "VERTICA_PORT": "5433",
        "VERTICA_DATABASE": "your_database",
        "VERTICA_USER": "your_username",
        "VERTICA_PASSWORD": "your_password"
      }
    }
  }
}
```

4. **Docker Configuration Options**

**Option A — Docker Compose (recommended for containerized environments)**

```json
{
  "mcpServers": {
    "vertica-mcp-stdio": {
      "command": "docker",
      "args": ["compose", 
               "-f", 
               "/path/to/vertica-mcp/docker-compose.yml", 
               "run", 
               "--rm", 
               "-T", 
               "mcp-stdio"]
    }
  }
}
```

**Option B — Direct `docker run` command**

```json
{
  "mcpServers": {
    "vertica-mcp-stdio": {
      "command": "docker",
      "args": ["run", 
               "-i", 
               "--rm", 
               "--env-file", 
               "/path/to/vertica-mcp/.env", 
               "vertica-mcp:latest"]
    }
  }
}
```

5. **Remote Transport Configuration (HTTP/SSE) via `mcp-remote`**

For remote deployments or when you prefer HTTP-based communication:

```json
{
  "mcpServers": {
    "vertica-mcp-http": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8000/mcp"]
    },
    "vertica-mcp-sse": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8000/sse"]
    }
  }
}
```

6. **Final Step: Restart Claude Desktop** 

After configuring, completely restart Claude Desktop and look for the (+) indicator which shows that the MCP server is connected and ready to use.

</details>

<details>
<summary><b>VS Code Integration</b></summary>

1. **Install GitHub Copilot Chat Extension**
   Ensure you have the latest version of the GitHub Copilot Chat extension installed in VS Code.

2. **Create MCP Configuration File**
   Create `.vscode/mcp.json` in your workspace root:

```json
{
  "servers": {
    "vertica-mcp": {
      "type": "http",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

3. **Enable MCP in VS Code Settings**
   Add these settings to your VS Code configuration:
```json
{
  "chat.mcp.enabled": true,
  "chat.mcp.discovery.enabled": true
}
```

</details>

<details>
<summary><b>Cursor IDE Integration</b></summary>

1. **Create MCP Configuration File**
   Create `mcp.json` in your Cursor configuration directory:
   - **Global Configuration**: `~/.cursor/mcp.json` (macOS/Linux) or `%UserProfile%\.cursor\mcp.json` (Windows)
   - **Per Project Configuration**: `<project>/.cursor/mcp.json`

```json
{
  "mcpServers": {
    "vertica-mcp": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8000/mcp"]
    }
  }
}
```

2. **Restart Cursor IDE**
   Completely restart Cursor and check the Available Tools section to verify the integration is working.

</details>

---

## CLI Reference

```bash
vertica-mcp [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-v, --verbose` | Increase verbosity level (-v, -vv, -vvv) | ERROR |
| `--env-file PATH` | Path to environment configuration file | `.env` |
| `--transport TYPE` | Transport protocol (stdio, sse, http) | `stdio` |
| `--port INT` | Port for SSE/HTTP transport | `8000` |
| `--host HOST` | Vertica database host | from env |
| `--bind-host HOST` | Host to bind SSE/HTTP server | `localhost` |
| `--db-port INT` | Vertica database port | from env |
| `--database NAME` | Database name | from env |
| `--user USERNAME` | Database username | from env |
| `--password PASS` | Database password | from env |
| `--connection-limit INT` | Maximum connections in pool | `10` |
| `--ssl` | Enable SSL for database connection | `false` |
| `--ssl-reject-unauthorized` | Reject unauthorized SSL certificates | `true` |
| `--http-path PATH` | Endpoint path for HTTP transport | `/mcp` |
| `--http-json` | Prefer batch JSON responses | `false` |
| `--http-stateless` | Use stateless HTTP sessions | `true` |

---

## Usage Examples

### Natural Language Queries

<table>
<tr>
<td>

**Basic Database Operations**
```text
"Show me all tables in the public schema"
"What's the structure of the customers table?"
"Get the last 100 orders from today"
"List all projections for the orders table"
"Get database status and health metrics"
```

</td>
<td>

**Performance Analysis and Monitoring**
```text
"Profile this query and suggest optimizations"
"Show system performance for the last hour"
"Find tables with high ROS container counts"
"Analyze the performance of this query: SELECT * FROM sales.orders WHERE order_date > '2024-01-01'"
"Monitor system performance for the last 30 minutes"
```

</td>
</tr>
<tr>
<td>

**Complex Analytics Queries**
```text
"Analyze sales trends by region and product"
"Find anomalies in transaction patterns"
"Generate a monthly revenue report"
"Execute this query safely: SELECT COUNT(*) FROM large_table"
```

</td>
<td>

**Database Management Tasks**
```text
"Check database health and storage usage"
"Monitor resource pool utilization"
"Identify and fix slow queries"
"Show me the current resource pool utilization"
```

</td>
</tr>
</table>

---

### Transport Options

| Transport | Use Case | Configuration |
|-----------|----------|---------------|
| **STDIO** | Local Claude Desktop integration | Default option, no network configuration required |
| **HTTP** | Remote deployments and cloud environments | RESTful API on custom port with JSON-RPC protocol |
| **SSE** | Real-time streaming applications | Server-sent events for live data updates |

---

## Testing & Validation

### Quick Health Check

Verify your database connection and server configuration with this simple test:

```bash
# Test database connection
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
from vertica_mcp.connection import VerticaConfig, VerticaConnectionManager

config = VerticaConfig.from_env()
manager = VerticaConnectionManager()
manager.initialize_default(config)
conn = manager.get_connection()
cursor = conn.cursor()
cursor.execute('SELECT version()')
print('Connected successfully to:', cursor.fetchone()[0])
manager.release_connection(conn)
"
```

### MCP Inspector Testing

The MCP Inspector provides comprehensive testing and validation capabilities:

```bash
# Install MCP Inspector
npm install -g @modelcontextprotocol/inspector

# Test local STDIO server
npx @modelcontextprotocol/inspector vertica_mcp/server.py

# Test HTTP server
npx @modelcontextprotocol/inspector http://localhost:8000/mcp
```

**MCP Inspector Configuration:**

Set the Transport Type to match your server configuration:

- **STDIO Transport Testing**
  - Command: `uv`
  - Arguments: `run --with mcp --with starlette --with uvicorn --with pydantic --with vertica-python mcp run vertica_mcp/server.py`

- **SSE Transport Testing**
  - URL: `http://localhost:8000/sse`

- **HTTP Transport Testing**
  - URL: `http://localhost:8000/mcp`

### API Endpoint Validation

Test your HTTP server endpoints directly with curl commands:

```bash
# Test tools list endpoint
curl -s http://localhost:8000/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'

# Test server initialization
curl -s http://localhost:8000/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"0.1.0","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}'
```

---

## Advanced Features

### Performance Optimization

<details>
<summary><b>Query Profiling & Optimization</b></summary>

The server automatically profiles queries and provides comprehensive optimization recommendations:

**Automatic Query Analysis:**
- Execution plan analysis with detailed step-by-step breakdown
- Join strategy recommendations based on table statistics
- Projection optimization suggestions for improved performance
- ROS container health monitoring and segmentation analysis

**Example Usage:**
```python
# Automatic query optimization with detailed feedback
"Profile and optimize: SELECT * FROM large_table JOIN dimension_table"
# Returns: Execution plan, identified bottlenecks, and CREATE PROJECTION statements
```

**Key Features:**
- Real-time performance metrics during query execution
- Historical performance comparison
- Automatic detection of inefficient patterns
- Specific recommendations for index creation and query rewriting

</details>

### Enterprise Integration

<details>
<summary><b>Production Deployment Checklist</b></summary>

Ensure your production deployment meets enterprise standards:

**Security Configuration:**
- [ ] Configure SSL/TLS for all database connections
- [ ] Set appropriate connection pool limits based on workload
- [ ] Enable read-only mode for production environments
- [ ] Configure schema-specific permissions for different user roles
- [ ] Implement proper authentication mechanisms

**Monitoring and Maintenance:**
- [ ] Set up comprehensive monitoring and alerting systems
- [ ] Implement rate limiting to prevent resource exhaustion
- [ ] Configure log rotation and retention policies
- [ ] Set up backup MCP servers for high availability
- [ ] Establish disaster recovery procedures

**Performance Optimization:**
- [ ] Tune connection pool parameters for your workload
- [ ] Configure appropriate query timeouts
- [ ] Set up caching strategies for frequently accessed data
- [ ] Monitor and optimize resource usage patterns

</details>

---

## Security Configuration

### Permission Management Levels

The Vertica MCP Server implements a comprehensive three-tier permission system:

1. **Global Permissions**: Control operations across all schemas and tables
2. **Schema-specific Permissions**: Fine-grained control per individual schema
3. **Connection Security**: SSL/TLS encryption and authentication options

### Security Best Practices

**Database Access Security:**
- **Use read-only credentials** for production deployments to minimize risk
- **Enable SSL/TLS encryption** for all database connections
- **Implement least-privilege access** with minimal required permissions
- **Use environment variables** instead of hardcoded credentials

**Network and Infrastructure Security:**
- **Restrict network access** using firewall rules and security groups
- **Monitor access logs** for suspicious activity and unauthorized attempts
- **Implement connection rate limiting** to prevent abuse
- **Regular security audits** of configuration and access patterns

**Operational Security:**
- **Regular credential rotation** following your organization's security policies
- **Audit trail maintenance** for all database operations
- **Secure backup procedures** for configuration and credentials
- **Incident response procedures** for security events

---

## Troubleshooting

### Common Issues and Solutions

#### Database Connection Problems

**Test Basic Connectivity:**
```bash
# Test network connectivity to Vertica server
telnet your_vertica_host 5433

# Test database credentials and permissions
vsql -h your_host -U your_user -d your_database
```

**Common Connection Issues:**
- **Network connectivity**: Verify firewall rules and network routing
- **Authentication failures**: Check username, password, and database permissions
- **SSL configuration**: Ensure SSL settings match server requirements
- **Connection pool exhaustion**: Monitor and adjust connection limits

#### MCP Client Integration Issues

**Troubleshooting Steps:**
1. **Complete client restart**: Fully restart the client application (Claude Desktop, VS Code, etc.)
2. **Configuration validation**: Verify JSON syntax in all configuration files
3. **Server log analysis**: Check server logs using `-vvv` verbose flag
4. **Isolation testing**: Test with MCP Inspector before client integration

**Common Configuration Problems:**
- Incorrect file paths in configuration
- Missing environment variables
- Port conflicts with other services
- Permission issues with executable files

#### Docker Deployment Issues

**Container Troubleshooting:**
```bash
# Check container logs for errors
docker logs vertica-mcp

# Test container internal connectivity
docker exec -it vertica-mcp curl http://localhost:8000/mcp

# Verify environment variable loading
docker exec -it vertica-mcp env | grep VERTICA
```

**Common Docker Issues:**
- Environment file not properly mounted
- Port mapping conflicts
- Network connectivity between containers
- Volume mounting permission problems

### Debug Mode and Logging

**Enable Maximum Verbosity:**
```bash
# Maximum verbosity for troubleshooting
vertica-mcp --transport http -vvv

# Log output to file for analysis
vertica-mcp --transport http -vv 2> debug.log
```

**Log Analysis Tips:**
- Look for connection establishment messages
- Check for permission denial errors
- Monitor query execution timestamps
- Identify resource exhaustion warnings

---

## Project Structure

```
vertica-mcp/
├── vertica_mcp/                 # Python package source code
│   ├── __init__.py             # Package initialization
│   ├── cli.py                  # Command-line interface implementation
│   ├── server.py               # Main MCP server implementation
│   ├── connection.py           # Database connection management
│   └── utils.py                # Utility functions and helpers
│
├── pyproject.toml               # Build configuration and metadata (PEP 621)
├── README.md                    # Project documentation
├── CHANGELOG.md                 # Release notes and version history
├── LICENSE                      # Apache 2.0 license
├── .gitignore                   # Git ignore rules
├── .dockerignore                # Docker ignore rules
├── .env.example                 # Sample environment file (do NOT commit .env)
│
├── docker-compose.yml           # Docker Compose configuration
├── docker-entrypoint.sh         # Docker container entry script
└── dockerfile                   # Docker image definition
```

---

## Contributing

We welcome and encourage contributions from the community! Please see our [Contributing Guide](CONTRIBUTING.md) for detailed information on how to get involved.

### Development Environment Setup

Set up your local development environment with these steps:

```bash
# Clone and setup development environment
git clone https://github.com/zaboura/vertica-mcp.git
cd vertica-mcp
uv sync

# Install development dependencies including testing and linting tools
uv pip install -e ".[dev]"

# Run comprehensive test suite
pytest tests/

# Code formatting and style checks
black vertica_mcp/
isort vertica_mcp/

# Type checking and static analysis
mypy vertica_mcp/
```

### Adding New Tools and Features

**When implementing new tools, follow these guidelines:**

1. **Tool Function Implementation**: Add tool functions in `server.py` with proper `@mcp.tool()` decorator and comprehensive docstrings
2. **Permission Management**: Implement appropriate permission checks using the connection manager
3. **Error Handling**: Add comprehensive error handling with informative error messages and proper logging
4. **Testing**: Write unit tests and integration tests for new functionality
5. **Documentation**: Update documentation including README, docstrings, and usage examples

### Contribution Guidelines

**Getting Started with Contributions:**

1. **Fork the repository** and create your feature branch from `main`
2. **Create a feature branch** with a descriptive name: `git checkout -b feature/AmazingFeature`
3. **Make your changes** following the existing code style and conventions
4. **Add tests** for any new functionality to ensure reliability
5. **Update documentation** as needed for new features or changes
6. **Commit your changes** with clear, descriptive messages: `git commit -m 'Add some AmazingFeature'`
7. **Push to your branch**: `git push origin feature/AmazingFeature`
8. **Open a Pull Request** with a detailed description of your changes

**Code Quality Standards:**
- Follow existing code style and formatting conventions
- Include comprehensive type hints for all functions
- Write clear, descriptive commit messages
- Ensure all tests pass before submitting pull requests
- Update documentation for any user-facing changes

---

## Community & Support

### Getting Help and Support

- **[GitHub Issues](https://github.com/zaboura/vertica-mcp/issues)** - Report bugs and request new features
- **[Discussions](https://github.com/zaboura/vertica-mcp/discussions)** - Ask questions and get community support
- **[Discord](https://discord.gg/vertica-mcp)** - Real-time chat with the community and maintainers
- **[Stack Overflow](https://stackoverflow.com/questions/tagged/vertica-mcp)** - Technical questions and answers

### Community Guidelines

**When seeking help or contributing:**
- Search existing issues and discussions before creating new ones
- Provide detailed information about your environment and configuration
- Include relevant error messages and log outputs
- Be respectful and constructive in all interactions
- Help others when you can share your knowledge and experience

---

## Resources

### Official Documentation and References

- **[MCP Specification](https://spec.modelcontextprotocol.io/)** - Complete protocol standard and implementation guidelines
- **[Vertica Documentation](https://www.vertica.com/docs/)** - Comprehensive database reference and best practices
- **[FastMCP Framework](https://github.com/modelcontextprotocol/fastmcp)** - Server framework documentation and examples
- **[Claude Desktop Guide](https://modelcontextprotocol.io/quickstart/user)** - Client setup and configuration instructions

### Learning Resources

**Understanding MCP:**
- Model Context Protocol introduction and concepts
- Best practices for MCP server development
- Security considerations for AI integrations

**Vertica Integration:**
- Database optimization techniques
- Performance tuning for analytics workloads
- Advanced query optimization strategies

**AI and Database Integration:**
- Natural language to SQL conversion techniques
- Database security in AI applications
- Performance monitoring for AI-driven queries

---

## Changelog

### Version 0.1.0 (2025-08-20) - Initial Release

**Core Features Implemented:**
- 11 comprehensive database tools for complete database interaction
- 5 AI-optimized prompts for enhanced user experience
- Support for STDIO, HTTP, and SSE transport protocols
- Docker support with complete compose configurations
- Enterprise-grade security features and permission management

**Key Capabilities:**
- Full schema exploration and metadata access
- Query execution with safety guards and optimization
- Real-time performance monitoring and analysis
- Comprehensive error handling and logging
- Production-ready deployment options

**Technical Achievements:**
- Connection pooling for optimal resource management
- Automatic query optimization and suggestion engine
- Multi-transport support for flexible deployment scenarios
- Comprehensive testing suite and validation tools

For complete version history and detailed changes, see [CHANGELOG.md](CHANGELOG.md)

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for complete terms and conditions.

**License Summary:**
- Commercial and non-commercial use permitted
- Modification and distribution allowed
- Patent protection included
- Warranty and liability disclaimers apply

---

## Acknowledgments

### Project Recognition

**Core Technologies:**
- **[Anthropic](https://www.anthropic.com/)** for creating and maintaining the Model Context Protocol standard
- **[Vertica](https://www.vertica.com/)** for providing the powerful analytics platform that makes this integration possible
- **[FastMCP](https://github.com/modelcontextprotocol/fastmcp)** for the excellent framework that simplified server development

<!-- **Community Contributions:**
- **The MCP Community** for continuous support, feedback, and contributions to the ecosystem
- **Contributors and Testers** who helped identify issues and improve functionality
- **Documentation Contributors** who helped make this project accessible to users -->

<!-- ### Special Thanks

Thanks to all the developers, database administrators, and AI enthusiasts who provided feedback during development and helped shape this project into a production-ready solution. -->

---

<p align="center">
<strong>Built with dedication for the AI and Database community</strong>
</p>