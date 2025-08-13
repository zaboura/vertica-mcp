# Vertica MCP Server

A Multi-Cloud Platform (MCP) server for Vertica database operations, providing AI-powered database management and query capabilities.

## Features

- **Comprehensive Database Operations**: Execute, stream, and analyze SQL queries
- **Schema Management**: List tables, views, projections, and schemas
- **Query Analysis**: Explain and profile query execution plans
- **Health Monitoring**: Database status and performance metrics
- **Projection Optimization**: Analyze and suggest projection improvements
- **AI Prompts**: Built-in prompts for query analysis and optimization
- **Connection Pooling**: Efficient connection management
- **Security**: Configurable operation permissions per schema

## Quick Start

### Option 1: Using MCP Dev Command (Recommended for Development)

The easiest way to run the MCP server with SSE transport is using the `mcp dev` command:

#### Prerequisites
- Python 3.12+
- MCP CLI installed: `pip install mcp[cli]`
- Vertica database running

#### 1. Install the Package
```bash
# Install in development mode
pip install -e .

# Or using uv
uv sync
```

#### 2. Configure Database Connection
The server will automatically load configuration from your existing `.env` file. Make sure it contains your Vertica connection details:
```bash
VERTICA_HOST=localhost
VERTICA_PORT=5433
VERTICA_DATABASE=your_database
VERTICA_USER=your_username
VERTICA_PASSWORD=your_password
VERTICA_CONNECTION_LIMIT=10
VERTICA_SSL=false
VERTICA_SSL_REJECT_UNAUTHORIZED=true
```

#### 3. Run with MCP Dev Command
```bash
# Run with SSE transport on port 8000
mcp dev mcp_config.json

# Or run directly with CLI
python vertica_mcp/cli.py --transport sse --port 8000
```

#### 4. Test the Connection
```bash
# Health check
curl http://localhost:8000/health

# Test SSE endpoint
curl -N http://localhost:8000/mcp
```

### Option 2: Using Docker (Recommended for Production)

#### Prerequisites

- Docker and Docker Compose installed
- At least 4GB RAM available for Vertica

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd vertica-mcp
```

### 2. Run with Docker Compose

#### Development Environment
```bash
# Start development stack with sample data
make dev

# Or manually:
docker-compose -f docker-compose.dev.yml up -d
```

#### Production Environment
```bash
# Start production stack
make run

# Or manually:
docker-compose up -d
```

### 3. Verify Installation

```bash
# Check health
make health

# Test database connection
make db-test

# View logs
make logs-dev  # for development
make logs      # for production
```

### 4. Access Services

- **MCP Server**: http://localhost:8000
- **Health Check**: http://localhost:8000/health
- **Adminer (Dev)**: http://localhost:8080 (Database management UI)
- **Nginx Proxy**: http://localhost:8080 (Production)

## Configuration

### MCP Configuration

The `mcp_config.json` file defines how the MCP server should be started. It uses your existing `.env` file for configuration:

```json
{
  "mcpServers": {
    "vertica-mcp": {
      "command": "python",
      "args": ["vertica_mcp/cli.py", "--transport", "sse", "--port", "8000"]
    }
  }
}
```

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# Database Connection
VERTICA_HOST=vertica
VERTICA_PORT=5433
VERTICA_DATABASE=VMart
VERTICA_USER=dbadmin
VERTICA_PASSWORD=password

# Connection Pool
VERTICA_CONNECTION_LIMIT=10

# SSL Settings
VERTICA_SSL=false
VERTICA_SSL_REJECT_UNAUTHORIZED=true

# Operation Permissions
ALLOW_INSERT_OPERATION=true
ALLOW_UPDATE_OPERATION=true
ALLOW_DELETE_OPERATION=true
ALLOW_DDL_OPERATION=true

# Schema-specific Permissions (optional)
SCHEMA_INSERT_PERMISSIONS=public:true,sales:true
SCHEMA_UPDATE_PERMISSIONS=public:true,sales:false
SCHEMA_DELETE_PERMISSIONS=public:false,sales:false
SCHEMA_DDL_PERMISSIONS=public:true,sales:false
```

### Docker Compose Overrides

For different environments, create override files:

```bash
# Development
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
```

## Available Tools

### Database Operations
- `execute_query`: Execute SQL queries
- `stream_query`: Stream large result sets
- `copy_data`: Bulk data insertion
- `database_status`: Get database health and metrics

### Schema Management
- `list_tables`: List tables in a schema
- `list_views`: List views in a schema
- `list_schemas`: List all schemas
- `list_projections`: List projections for a table
- `get_table_structure`: Get detailed table structure

### Query Analysis
- `explain_query`: Get query execution plan
- `profile_query`: Profile query performance
- `get_long_running_queries`: Monitor long-running queries
- `analyze_projection_optimization`: Analyze projection performance

### Health Monitoring
- `check_vertica_health`: Comprehensive health check
- `get_vertica_metrics`: System and performance metrics

### AI Prompts
- `analyze_query_performance`: AI prompt for query analysis
- `write_query_for_task`: AI prompt for query generation
- `database_status_prompt`: AI prompt for status reporting

## Development

### Local Development

```bash
# Install dependencies
make install

# Run code quality checks
make check

# Format code
make format

# Run tests
make test
```

### Docker Development

```bash
# Build development image
make build-dev

# Start development environment
make dev

# Access container shell
make shell-dev

# View logs
make logs-dev
```

### Database Initialization

The development environment includes sample data:

```sql
-- Sample schemas: sales, analytics, staging
-- Sample tables: customers, orders, products, daily_sales
-- Sample data: 3 customers, 4 products, 3 orders
```

## API Usage

### SSE Transport

```bash
# Connect to MCP server
curl -N http://localhost:8000/

# Health check
curl http://localhost:8000/health
```

### Example Queries

```python
# List tables in sales schema
await execute_query("SELECT table_name FROM v_catalog.tables WHERE table_schema = 'sales'")

# Get customer orders
await execute_query("SELECT * FROM sales.customer_orders")

# Analyze query performance
await explain_query("SELECT * FROM sales.orders WHERE total_amount > 100")

# Check database health
await check_vertica_health()
```

## Troubleshooting

### Common Issues

1. **Vertica not starting**: Ensure sufficient RAM (4GB+)
2. **Connection refused**: Wait for Vertica to fully initialize
3. **Permission denied**: Check schema permissions in environment variables

### Debug Commands

```bash
# Check container status
docker-compose ps

# View detailed logs
docker-compose logs vertica-mcp

# Test database connectivity
make db-test

# Access database directly
docker-compose exec vertica /opt/vertica/bin/vsql -U dbadmin -d VMart
```

### Health Checks

```bash
# Service health
make health

# Database health
curl http://localhost:8000/health

# Container health
docker-compose ps
```

## Production Deployment

### Security Considerations

1. **Change default passwords**
2. **Enable SSL/TLS**
3. **Restrict operation permissions**
4. **Use secrets management**
5. **Enable logging and monitoring**

### Scaling

```bash
# Scale MCP server
docker-compose up -d --scale vertica-mcp=3

# Use load balancer
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and checks: `make check`
5. Submit a pull request

## License

[Your License Here]

## Support

For issues and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the logs: `make logs`


