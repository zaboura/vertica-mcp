# Vertica MCP Server

<p align="center">
  <img src="https://img.shields.io/badge/MCP-1.0-blue.svg" alt="MCP Version">
  <img src="https://img.shields.io/badge/Python-3.11+-green.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Vertica-24.x+-orange.svg" alt="Vertica Version">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

A powerful Model Context Protocol (MCP) server for Vertica Analytics Database, enabling AI assistants like Claude to directly interact with your Vertica database through standardized tools and prompts.

## 🎯 Overview

The Vertica MCP Server bridges the gap between AI language models and your Vertica database, providing secure, controlled access to database operations through the Model Context Protocol. This enables Claude and other MCP-compatible AI assistants to query, analyze, and manage your Vertica databases with natural language commands.

### What is MCP?

The Model Context Protocol (MCP) is an open standard developed by Anthropic that provides a universal way for AI assistants to connect with external tools and data sources. Think of it as "USB-C for AI" - a standardized interface that allows any MCP-compatible AI to interact with your systems without custom integrations.

## ✨ Features

### Core Capabilities
- **🔍 Query Execution**: Execute SELECT queries with automatic pagination and safety controls
- **📊 Performance Analysis**: Profile queries and get execution plans with optimization recommendations
- **🗂️ Schema Management**: List and explore tables, views, projections, and schemas
- **💾 Streaming Support**: Handle large result sets with efficient streaming and pagination
- **🔒 Security**: Configurable operation permissions at global and schema levels
- **⚡ Connection Pooling**: Efficient connection management with configurable pool sizes
- **🎨 AI Prompts**: Built-in intelligent prompts for query optimization and database analysis

### Built-in Tools

| Tool | Description |
|------|-------------|
| `run_query_safely` | Execute queries with automatic large result set detection and confirmation |
| `execute_query_paginated` | Paginated query execution with LIMIT/OFFSET support |
| `execute_query_stream` | Stream large result sets in batches |
| `get_table_structure` | Get detailed table structure including columns and constraints |
| `get_schema_tables` | List all tables in a schema |
| `get_schema_views` | List all views in a schema |
| `get_database_schemas` | List all database schemas |
| `get_table_projections` | List projections for a specific table |
| `database_status` | Get database health and usage statistics |
| `profile_query` | Profile query performance and get execution plans |
| `analyze_system_performance` | Monitor system performance metrics and resource usage |

### AI-Powered Prompts

| Prompt | Purpose |
|--------|---------|
| `sql_query_safety_guard` | Prevents accidental large result sets |
| `vertica_performance_analyzer` | Deep-dive query performance analysis |
| `vertica_sql_assistant` | Generate optimized Vertica SQL queries |
| `vertica_health_dashboard` | Comprehensive database status visualization |
| `vertica_system_monitor` | Real-time performance monitoring |

## 📋 Prerequisites

- **Python** 3.11 or higher
- **Vertica Database** (accessible)
- **uv** (Python package manager) - [Installation guide](https://github.com/astral-sh/uv)
- **Claude Desktop** or another MCP-compatible client (optional)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/zaboura/vertica-mcp.git
cd vertica-mcp
```

### 2. Install uv (if not already installed)

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Or, from [PyPi](https://pypi.org/project/uv/)


```bash
pip install uv
```

### 3. Set Up Python Environment

```bash
# Create virtual environment and install dependencies
uv venv
uv pip install -e .

# Or install all dependencies at once
uv sync
```

### 4. Configure Database Connection

Create a `.env` file in the project root:

```bash
# Database Connection
VERTICA_HOST=localhost
VERTICA_PORT=5433
VERTICA_DATABASE=your_database
VERTICA_USER=your_username
VERTICA_PASSWORD=your_password

# Connection Pool
VERTICA_CONNECTION_LIMIT=10
VERTICA_LAZY_INIT=1  # Lazy connection initialization

# SSL Configuration
VERTICA_SSL=false
VERTICA_SSL_REJECT_UNAUTHORIZED=true

# Operation Permissions (optional)
ALLOW_INSERT_OPERATION=false
ALLOW_UPDATE_OPERATION=false
ALLOW_DELETE_OPERATION=false
ALLOW_DDL_OPERATION=false

# Schema-specific Permissions (optional)
SCHEMA_INSERT_PERMISSIONS=public:true,sales:false
SCHEMA_UPDATE_PERMISSIONS=public:true,sales:false
SCHEMA_DELETE_PERMISSIONS=public:false,sales:false
SCHEMA_DDL_PERMISSIONS=public:false,sales:false
```

## 🏃 Running the Server

### Option 1: Using CLI (Recommended)

The server supports multiple transport protocols:

#### STDIO Transport (Default)
```bash
# Basic usage
vertica-mcp

# With verbose logging
vertica-mcp -v

# With custom environment file
vertica-mcp --env-file production.env
```

#### SSE Transport (HTTP Server-Sent Events)
```bash
# Start SSE server on default port 8000
vertica-mcp --transport sse

# Custom port and host
vertica-mcp --transport sse --port 3000 --bind-host 0.0.0.0

# With database override
vertica-mcp --transport sse --host db.example.com --database production_db
```

### Option 2: Using Python Module

```bash
# STDIO transport
python -m vertica_mcp.cli

# SSE transport with options
python -m vertica_mcp.cli --transport sse --port 8000 -vv
```

### Option 3: Using uv run

```bash
# Run with uv directly
uv run vertica-mcp --transport sse --port 8000

# Or from the server module
uv run python vertica_mcp/server.py
```

## 🔍 Testing with MCP Inspector

MCP Inspector is a visual debugging tool for MCP servers. It's invaluable for testing your server implementation.

### Installation and Usage

```bash
# Install MCP Inspector globally
npm install -g @modelcontextprotocol/inspector

# Run inspector from your project directory
cd vertica-mcp
npx @modelcontextprotocol/inspector vertica_mcp/server.py

# The inspector UI will open at http://localhost:5173
```

### Using the Inspector

1. **Test Tools**: Click on "Tools" tab to see all available tools
2. **Execute Queries**: Test `run_query_safely` with sample queries
3. **View Responses**: See formatted JSON responses and errors
4. **Test Prompts**: Try the built-in AI prompts
5. **Monitor Performance**: Use `profile_query` to analyze query performance

## 🤖 Claude Desktop Integration

### Configuration

1. **Open Claude Desktop Settings**
   - On macOS: Claude menu → Settings
   - On Windows: File → Settings

2. **Edit Configuration**
   - Click "Edit Config" in Developer tab
   - This opens `claude_desktop_config.json`

3. **Add Server Configuration**

#### For STDIO Transport:
```json
{
  "mcpServers": {
    "vertica-mcp": {
      "command": "uv",
      "args": [
        "run",
        "--with", "mcp[cli]",
        "--with", "vertica-python",
        "--with", "python-dotenv",
        "--with", "pydantic",
        "--with", "starlette",
        "--with", "uvicorn",
        "mcp",
        "run",
        "/path/to/vertica-mcp/vertica_mcp/server.py"
      ],
      "env": {
        "PYTHONPATH": "/path/to/vertica-mcp",
        "VERTICA_HOST": "localhost",
        "VERTICA_PORT": "5433",
        "VERTICA_DATABASE": "your_database",
        "VERTICA_USER": "your_username",
        "VERTICA_PASSWORD": "your_password",
        "VERTICA_CONNECTION_LIMIT": "10"
      }
    }
  }
}
```

#### For SSE Transport:
```json
{
  "mcpServers": {
    "vertica-mcp-sse": {
      "command": "npx",
      "args": ["@modelcontextprotocol/remote", "http://localhost:8000/sse"]
    }
  }
}
```

4. **Restart Claude Desktop**
   - Quit and restart Claude Desktop
   - Look for the MCP indicator (🔌) in the chat input area
   - Click it to see available tools

### Usage Examples

Once connected, you can ask Claude:

```text
"Show me all tables in the public schema"
"Analyze the performance of this query: SELECT * FROM sales.orders WHERE order_date > '2024-01-01'"
"What's the current database status and usage?"
"Profile this query and suggest optimizations"
"Monitor system performance for the last 15 minutes"
```

## 🛠️ CLI Options Reference

```bash
vertica-mcp [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-v, --verbose` | Increase verbosity (use multiple times: -v, -vv, -vvv) | ERROR |
| `--env-file PATH` | Path to .env file | `.env` |
| `--transport TYPE` | Transport protocol (stdio, sse, http) | `stdio` |
| `--port INT` | Port for SSE/HTTP transport | `8000` |
| `--host HOST` | Vertica database host | `localhost` |
| `--bind-host HOST` | Host to bind SSE/HTTP server | `localhost` |
| `--db-port INT` | Vertica database port | `5433` |
| `--database NAME` | Database name | from env |
| `--user USERNAME` | Database username | from env |
| `--password PASS` | Database password | from env |
| `--connection-limit INT` | Max connections in pool | `10` |
| `--ssl` | Enable SSL for database | `false` |
| `--ssl-reject-unauthorized` | Reject unauthorized SSL certs | `true` |

## 🔧 Advanced Configuration

### Connection Pool Tuning

```bash
# Optimize for high concurrency
VERTICA_CONNECTION_LIMIT=50
VERTICA_LAZY_INIT=1  # Don't create connections until needed

# Optimize for low latency
VERTICA_CONNECTION_LIMIT=20
VERTICA_LAZY_INIT=0  # Pre-create all connections
```

### Security Configuration

#### Global Permissions
```bash
# Read-only mode
ALLOW_INSERT_OPERATION=false
ALLOW_UPDATE_OPERATION=false
ALLOW_DELETE_OPERATION=false
ALLOW_DDL_OPERATION=false
```

#### Schema-Specific Permissions
```bash
# Format: schema1:permission,schema2:permission
SCHEMA_INSERT_PERMISSIONS=staging:true,production:false
SCHEMA_UPDATE_PERMISSIONS=staging:true,production:false
SCHEMA_DELETE_PERMISSIONS=staging:true,production:false
SCHEMA_DDL_PERMISSIONS=staging:false,production:false
```

### SSL/TLS Configuration

```bash
# Enable SSL with certificate verification
VERTICA_SSL=true
VERTICA_SSL_REJECT_UNAUTHORIZED=true

# Enable SSL without certificate verification (development only)
VERTICA_SSL=true
VERTICA_SSL_REJECT_UNAUTHORIZED=false
```

## 📊 Using the Tools

### Query Execution with Safety

The `run_query_safely` tool prevents accidental large result sets:

```python
# Automatically detects large results
await run_query_safely(
    query="SELECT * FROM large_table",
    row_threshold=1000,  # Warn if >1000 rows
    proceed=False  # Don't proceed without confirmation
)
```

### Performance Profiling

```python
# Get detailed execution plan
await profile_query(
    query="SELECT * FROM sales.orders JOIN sales.customers ON ..."
)
```

### System Monitoring

```python
# Monitor system performance
await get_system_performance(
    window_minutes=15,
    bucket="minute",
    top_n=5
)
```

## 🧪 Development

### Project Structure

```
vertica-mcp/
├── vertica_mcp/
│   ├── __init__.py       # Package initialization
│   ├── cli.py            # CLI interface
│   ├── server.py         # MCP server implementation
│   ├── connection.py     # Database connection management
│   └── utils.py          # Utility functions
├── .env                  # Environment configuration
├── pyproject.toml        # Project configuration
├── setup.py             # Setup configuration
└── README.md            # This file
```

### Running Tests

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest --cov=vertica_mcp tests/
```

### Code Quality

```bash
# Format code
black vertica_mcp/
isort vertica_mcp/

# Lint code
pylint vertica_mcp/
mypy vertica_mcp/
```

## 🐛 Troubleshooting

### Common Issues

#### Connection Refused
```bash
# Check Vertica is running
vsql -h localhost -U dbadmin -d your_database

# Test network connectivity
telnet localhost 5433
```

#### MCP Server Not Showing in Claude
1. Verify configuration in `claude_desktop_config.json`
2. Check server logs: `vertica-mcp -vv`
3. Restart Claude Desktop completely
4. Test with MCP Inspector first

#### Large Query Results
- Use `run_query_safely` with appropriate `row_threshold`
- Implement pagination with `query_page`
- Consider using `stream_query` for very large datasets

### Debug Mode

Enable detailed logging:

```bash
# Maximum verbosity
vertica-mcp -vvv

# Write logs to file
vertica-mcp -vv 2> debug.log
```

## 📚 Documentation

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [Vertica Python Client](https://github.com/vertica/vertica-python)
- [FastMCP Documentation](https://github.com/modelcontextprotocol/fastmcp)
- [Claude Desktop Guide](https://modelcontextprotocol.io/quickstart/user)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Anthropic](https://www.anthropic.com/) for developing the Model Context Protocol
- [Vertica](https://www.vertica.com/) for the powerful analytics database
- The MCP community for continuous improvements and support

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/zaboura/vertica-mcp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/zaboura/vertica-mcp/discussions)
- **Email**: support@your-domain.com

---

<p align="center">
Built with ❤️ for the AI and Database community
</p>