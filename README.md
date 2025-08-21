# Vertica MCP Server

<div align="center">

![Vertica MCP Banner](https://img.shields.io/badge/Vertica_MCP-Enterprise_Analytics-00A0E3?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdOb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTggMTZDMTIuNDE4MyAxNiAxNiAxMi40MTgzIDE2IDhDMTYgMy41ODE3MiAxMi40MTgzIDAgOCAwQzMuNTgxNzIgMCAwIDMuNTgxNzIgMCA4QzAgMTIuNDE4MyAzLjU4MTcyIDE2IDggMTZ6IiBmaWxsPSIjMDBBMEUzIi8+Cjwvc3ZnPg==)

**Transform your Vertica Analytics Database into an AI-powered intelligence layer**

[![MCP Version](https://img.shields.io/badge/MCP-2025--08--20-blue.svg)](https://spec.modelcontextprotocol.io/)
[![Python Version](https://img.shields.io/badge/Python-3.11+-green.svg)](https://www.python.org/)
[![Vertica Version](https://img.shields.io/badge/Vertica-24.x+-orange.svg)](https://www.vertica.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker)](https://hub.docker.com/)
[![CI/CD](https://img.shields.io/github/actions/workflow/status/zaboura/vertica-mcp/ci.yml?label=CI/CD)](https://github.com/zaboura/vertica-mcp/actions)

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-documentation) • [🎯 Features](#-features) • [🤝 Contributing](#-contributing) • [💬 Community](#-community)

</div>

---

## 🌟 Why Vertica MCP?

The **Vertica MCP Server** is a production-ready implementation of the [Model Context Protocol](https://modelcontextprotocol.io/) that transforms your Vertica Analytics Database into an intelligent, AI-accessible data platform. Built with enterprise security and performance in mind, it enables AI assistants like Claude, ChatGPT, and Cursor to directly query, analyze, and optimize your Vertica databases through natural language.

### What is MCP?

The Model Context Protocol (MCP) is an open standard developed by Anthropic that provides a universal way for AI assistants to connect with external tools and data sources. Think of it as "USB-C for AI" - a standardized interface that allows any MCP-compatible AI to interact with your systems without custom integrations.

### 🎯 Key Benefits

- **🔌 Universal AI Connectivity**: Connect any MCP-compatible AI to your Vertica database without custom integrations
- **🛡️ Enterprise Security**: Fine-grained permissions at schema and operation levels with SSL/TLS support
- **⚡ High Performance**: Connection pooling, query streaming, and automatic pagination for handling massive datasets
- **🧠 AI-Optimized**: Built-in prompts and tools specifically designed for database analysis and optimization
- **🔄 Multiple Transports**: Support for STDIO, HTTP, and SSE to fit any deployment scenario
- **📊 Production Ready**: Battle-tested with comprehensive error handling, logging, and monitoring

---

## 📋 Prerequisites

- **Python** 3.11 or higher
- **Vertica Database** (accessible instance)
- **uv** (Python package manager) - [Installation guide](https://github.com/astral-sh/uv)
- **Docker** (optional, for containerized deployment)
- **Claude Desktop** or another MCP-compatible client

---

## 🚀 Quick Start

### Method 1: Local Installation (Recommended for Development)

```bash
# 1. Clone the repository
git clone https://github.com/zaboura/vertica-mcp.git
cd vertica-mcp

# 2. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Setup environment
uv sync
source .venv/bin/activate

# 4. Install in development mode
uv pip install -e .

# 5. Configure database connection
cp .env.example .env
# Edit .env with your Vertica credentials

# 6. Run the server
vertica-mcp  # STDIO for Claude Desktop
# or
vertica-mcp --transport http --port 8000 --bind-host 0.0.0.0  # HTTP for remote access
```

### Method 2: Docker Deployment

#### Build Docker Image

```bash
# Build from Dockerfile
docker build -t vertica-mcp:latest .

# Or use docker-compose
docker-compose build
```

#### Run with Docker Compose

```bash
# STDIO transport
docker-compose up mcp-stdio

# HTTP transport
docker-compose up mcp-http

# SSE transport  
docker-compose up mcp-sse
```

#### Manual Docker Run

```bash
# HTTP transport
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

## 🎯 Features

### 🔧 Core Tools

<table>
<tr>
<td width="50%">

#### Query Execution
- `run_query_safely` - Smart query execution with large result detection
- `execute_query_paginated` - Efficient pagination for large datasets
- `execute_query_stream` - Real-time streaming for massive results

</td>
<td width="50%">

#### Schema Management
- `get_database_schemas` - Explore database organization
- `get_schema_tables` - List tables with metadata
- `get_table_structure` - Detailed column and constraint info
- `get_table_projections` - Vertica-specific projection analysis
- `get_schema_views` - List all views in a schema

</td>
</tr>
<tr>
<td width="50%">

#### Performance Analysis
- `profile_query` - Query execution plans and optimization
- `analyze_system_performance` - Real-time resource monitoring
- `database_status` - Health metrics and usage statistics

</td>
<td width="50%">

#### AI-Powered Prompts
- 🛡️ **SQL Safety Guard** - Prevents accidental large queries
- 🚀 **Performance Analyzer** - Deep query optimization
- 💡 **SQL Assistant** - Intelligent query generation
- 📊 **Health Dashboard** - Visual database insights
- ⚡ **System Monitor** - Real-time performance tracking

</td>
</tr>
</table>

### 🛡️ Security Features

- **Multi-Level Permissions**: Global and schema-specific access controls
- **SSL/TLS Encryption**: Secure database connections
- **Connection Pooling**: Efficient resource management with configurable limits
- **Read-Only Mode**: Default safe configuration for production
- **OAuth Support**: Enterprise authentication for remote deployments

---

## 📖 Documentation

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

# Connection Pool (Optional)
VERTICA_CONNECTION_LIMIT=10
VERTICA_LAZY_INIT=1  # Delay connection until first use

# SSL Configuration (Optional)
VERTICA_SSL=false
VERTICA_SSL_REJECT_UNAUTHORIZED=true

# Security Permissions (Optional - defaults to read-only)
ALLOW_INSERT_OPERATION=false
ALLOW_UPDATE_OPERATION=false
ALLOW_DELETE_OPERATION=false
ALLOW_DDL_OPERATION=false

# Schema-specific Permissions (Optional)
SCHEMA_INSERT_PERMISSIONS=staging:true,production:false
SCHEMA_UPDATE_PERMISSIONS=staging:true,production:false
SCHEMA_DELETE_PERMISSIONS=staging:false,production:false
SCHEMA_DDL_PERMISSIONS=staging:false,production:false
```

</details>

### Client Integration

<details>
<summary><b>Claude Desktop</b></summary>

1. **Locate Configuration File**
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

2. **Add Server Configuration**

**For local STDIO:**

```json
{
  "mcpServers": {
    "vertica-mcp-stdio": {
      "command": "uv",
      "args": [
        "run",
        "vertica-mcp"
      ],
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

**Option A — Docker Compose (recommended)**

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

**Option B — Direct docker run (alternative)**

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

**For Remote HTTP Server (works for docker):**

```json
{
  "mcpServers": {
    "vertica-mcp-http": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8000/mcp"]
    }
  }
}
```

**For Remote SSE Server (works for docker):**

```json
{
  "mcpServers": {
    "vertica-mcp-sse": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8000/sse"]
    }
  }
}
```

3. **Restart Claude Desktop** and look for the 🔌 indicator

</details>

<details>
<summary><b>VS Code</b></summary>

1. **Install GitHub Copilot Chat Extension**
2. **Create `.vscode/mcp.json`** in your workspace:

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

3. **Enable MCP in Settings**:
```json
{
  "chat.mcp.enabled": true,
  "chat.mcp.discovery.enabled": true
}
```

</details>

<details>
<summary><b>Cursor</b></summary>

1. **Create `mcp.json`** in your Cursor config directory:
   - **Global**: `~/.cursor/mcp.json` (macOS/Linux) or `%UserProfile%\.cursor\mcp.json` (Windows)
   - **Per project**: `<project>/.cursor/mcp.json`

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

2. **Restart Cursor** and check Available Tools

</details>

---

## 🔧 CLI Reference

```bash
vertica-mcp [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-v, --verbose` | Increase verbosity (-v, -vv, -vvv) | ERROR |
| `--env-file PATH` | Path to .env file | `.env` |
| `--transport TYPE` | Transport protocol (stdio, sse, http) | `stdio` |
| `--port INT` | Port for SSE/HTTP transport | `8000` |
| `--host HOST` | Vertica database host | from env |
| `--bind-host HOST` | Host to bind SSE/HTTP server | `localhost` |
| `--db-port INT` | Vertica database port | from env |
| `--database NAME` | Database name | from env |
| `--user USERNAME` | Database username | from env |
| `--password PASS` | Database password | from env |
| `--connection-limit INT` | Max connections in pool | `10` |
| `--ssl` | Enable SSL for database | `false` |
| `--ssl-reject-unauthorized` | Reject unauthorized SSL certs | `true` |
| `--http-path PATH` | Endpoint path for HTTP | `/mcp` |
| `--http-json` | Prefer batch JSON responses | `false` |
| `--http-stateless` | Use stateless HTTP sessions | `true` |

---

## 🎮 Usage Examples

### Natural Language Queries

<table>
<tr>
<td>

**Basic Operations**
```text
"Show me all tables in the public schema"
"What's the structure of the customers table?"
"Get the last 100 orders from today"
"List all projections for the orders table"
"Get database status and health metrics"
```

</td>
<td>

**Performance Analysis**
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

**Complex Analytics**
```text
"Analyze sales trends by region and product"
"Find anomalies in transaction patterns"
"Generate a monthly revenue report"
"Execute this query safely: SELECT COUNT(*) FROM large_table"
```

</td>
<td>

**Database Management**
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
| **STDIO** | Local Claude Desktop | Default, no network required |
| **HTTP** | Remote/Cloud deployments | RESTful API on custom port |
| **SSE** | Real-time streaming | Server-sent events for live data |

---

## 🧪 Testing & Validation

### Quick Health Check

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
print('✅ Connected to:', cursor.fetchone()[0])
manager.release_connection(conn)
"
```

### MCP Inspector

```bash
# Install MCP Inspector
npm install -g @modelcontextprotocol/inspector

# Test local server
npx @modelcontextprotocol/inspector vertica_mcp/server.py

# Test HTTP server
npx @modelcontextprotocol/inspector http://localhost:8000/mcp
```

Use the MCP Inspector and set Transport Type to match your server, then configure:

- **STDIO**
  * Command: uv
  * Arguments: `run --with mcp --with starlette --with uvicorn --with pydantic --with vertica-python mcp run vertica_mcp/server.py`

- **SSE**
  * URL: `http://localhost:8000/sse`

- **HTTP**
  * URL: `http://localhost:8000/mcp`

### Validate API Endpoints

```bash
# Test tools list
curl -s http://localhost:8000/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'

# Test initialization
curl -s http://localhost:8000/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"0.1.0","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}'
```

---

## 🚀 Advanced Features

### Performance Optimization

<details>
<summary><b>Query Profiling & Optimization</b></summary>

The server automatically profiles queries and provides:
- Execution plan analysis
- Join strategy recommendations
- Projection optimization suggestions
- ROS container health monitoring

Example:
```python
# Automatic query optimization
"Profile and optimize: SELECT * FROM large_table JOIN dimension_table"
# Returns: Execution plan, bottlenecks, and CREATE PROJECTION statements
```

</details>

### Enterprise Integration

<details>
<summary><b>Production Deployment Checklist</b></summary>

- [ ] Configure SSL/TLS for database connections
- [ ] Set appropriate connection pool limits
- [ ] Enable read-only mode for production
- [ ] Configure schema-specific permissions
- [ ] Set up monitoring and alerting
- [ ] Implement rate limiting
- [ ] Configure log rotation
- [ ] Set up backup MCP servers for HA

</details>

---

## 🛡️ Security Configuration

### Permission Levels

1. **Global Permissions**: Control operations across all schemas
2. **Schema-specific Permissions**: Fine-grained control per schema
3. **Connection Security**: SSL/TLS encryption options

### Best Practices

- **Use read-only credentials** for production deployments
- **Enable SSL** for database connections
- **Restrict network access** using firewall rules
- **Monitor logs** for suspicious activity
- **Use environment files** instead of hardcoded credentials

---

## 🐛 Troubleshooting

### Common Issues

#### Connection Problems
```bash
# Test database connectivity
telnet your_vertica_host 5433

# Check Vertica service
vsql -h your_host -U your_user -d your_database
```

#### MCP Client Issues
1. **Completely restart** the client application
2. **Verify JSON syntax** in configuration files
3. **Check server logs** with `-vvv` flag
4. **Test with MCP Inspector** first

#### Docker Issues
```bash
# Check container logs
docker logs vertica-mcp

# Test container connectivity
docker exec -it vertica-mcp curl http://localhost:8000/mcp
```

### Debug Mode

```bash
# Maximum verbosity
vertica-mcp --transport http -vvv

# Log to file
vertica-mcp --transport http -vv 2> debug.log
```

---

## 📁 Project Structure

```
vertica-mcp/
├── vertica_mcp/              # Main package
│   ├── __init__.py          # Package initialization
│   ├── cli.py               # Command-line interface
│   ├── server.py            # MCP server implementation
│   ├── connection.py        # Database connection management
│   └── utils.py            # Utility functions
├── docker-compose.yml       # Docker Compose configuration
├── docker-entrypoint.sh    # Docker entry point script
├── Dockerfile              # Docker image definition
├── pyproject.toml          # Project configuration
├── setup.py               # Setup script
├── .env.example           # Environment template
└── README.md             # This file
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/zaboura/vertica-mcp.git
cd vertica-mcp
uv sync

# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest tests/

# Code formatting
black vertica_mcp/
isort vertica_mcp/

# Type checking
mypy vertica_mcp/
```

### Adding New Tools

1. **Add tool function** in `server.py` with `@mcp.tool()` decorator
2. **Implement permission checks** using the connection manager
3. **Add comprehensive error handling** and logging
4. **Write tests** for the new functionality
5. **Update documentation**

### Contributing Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 💬 Community & Support

- **[GitHub Issues](https://github.com/zaboura/vertica-mcp/issues)** - Bug reports and feature requests
- **[Discussions](https://github.com/zaboura/vertica-mcp/discussions)** - Questions and community support
- **[Discord](https://discord.gg/vertica-mcp)** - Real-time chat with the community
- **[Stack Overflow](https://stackoverflow.com/questions/tagged/vertica-mcp)** - Technical Q&A

---

## 📚 Resources

### Official Documentation
- **[MCP Specification](https://spec.modelcontextprotocol.io/)** - Protocol standard
- **[Vertica Documentation](https://www.vertica.com/docs/)** - Database reference
- **[FastMCP Framework](https://github.com/modelcontextprotocol/fastmcp)** - Server framework
- **[Claude Desktop Guide](https://modelcontextprotocol.io/quickstart/user)** - Client setup instructions

<!-- ### Tutorials & Guides
- [Building Your First MCP Server](docs/tutorials/getting-started.md)
- [Advanced Query Optimization](docs/guides/optimization.md)
- [Security Best Practices](docs/guides/security.md)
- [Troubleshooting Guide](docs/troubleshooting.md) -->

---

## 🔄 Changelog

### v0.1.0 (2025-08-20)
- 🎉 Initial release with core functionality
- ✅ 11 database tools implemented
- 🧠 5 AI-optimized prompts
- 🚀 Support for STDIO, HTTP, and SSE transports
- 🐳 Docker support with compose configurations
- 🔐 Enterprise security features

[See full changelog](CHANGELOG.md)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[Anthropic](https://www.anthropic.com/)** for creating the Model Context Protocol
- **[Vertica](https://www.vertica.com/)** for the powerful analytics platform
- **[FastMCP](https://github.com/modelcontextprotocol/fastmcp)** for the excellent framework
- **The MCP Community** for continuous support and contributions

<!-- ---

<div align="center">

**Built with ❤️ for the AI and Database community**

[![Star History](https://api.star-history.com/svg?repos=zaboura/vertica-mcp&type=Date)](https://star-history.com/#zaboura/vertica-mcp&Date)

[⬆ Back to top](#vertica-mcp-server)

</div> -->

---

<p align="center">
<strong>Built with ❤️ for the AI and Database community</strong>
</p>