# Changelog

All notable changes to the Vertica MCP Server project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- [ ] Migration from HTTP+SSE to Streamable HTTP transport
- [ ] Enhanced security considerations

---

## [0.1.0] - 2025-08-21

### 🎉 Initial Release

The first stable release of the Vertica MCP Server, providing enterprise-grade AI connectivity to Vertica Analytics Database through the Model Context Protocol.

### ✅ Core Database Tools

#### Query Execution & Management
- **`run_query_safely`** - Intelligent query execution with large result detection and user confirmation
  - Automatic result size probing with configurable thresholds (default: 1000 rows)
  - Safety guards against accidental large queries
  - Support for pagination and streaming modes
  - Query timeout protection (default: 10 minutes)
  - SQL injection prevention with pattern detection
- **`execute_query_paginated`** - Efficient pagination for large datasets
  - Configurable page sizes (default: 2000 rows)
  - Automatic offset management
  - Memory-efficient result handling
- **`execute_query_stream`** - Real-time streaming for massive results
  - Batched processing with configurable batch sizes
  - Memory usage monitoring and limits
  - Progressive result delivery

#### Schema & Metadata Management
- **`get_database_schemas`** - Database schema exploration with system/user schema classification
- **`get_schema_tables`** - Table listing with metadata caching (5-minute TTL)
- **`get_schema_views`** - View enumeration and definition access
- **`get_table_structure`** - Comprehensive table metadata including:
  - Column definitions with data types, constraints, and defaults
  - Primary/foreign key relationships
  - Nullability and constraint information
- **`get_table_projections`** - Vertica-specific projection analysis
  - Super projection identification
  - Projection type classification
  - Anchor table relationships

#### Performance & Monitoring
- **`profile_query`** - Advanced query profiling with execution plan analysis
  - Automatic query labeling with UUID tracking
  - Transaction and statement ID resolution
  - Execution plan extraction from `v_internal.dc_explain_plans`
  - Performance metrics collection
- **`analyze_system_performance`** - Real-time system monitoring
  - CPU and memory utilization tracking
  - Configurable time windows (default: 10 minutes)
  - Top resource-consuming tables identification
  - ROS container health monitoring
- **`database_status`** - Comprehensive health dashboard
  - License utilization tracking with capacity alerts
  - 7-day usage trend analysis
  - Cluster node status monitoring
  - Version and configuration reporting
- **`generate_health_dashboard`** - Consolidated metrics with multiple output formats
  - Compact, detailed, and JSON response formats
  - Token-optimized outputs for AI efficiency
  - Automated alert generation for critical thresholds

### 🧠 AI-Optimized Prompts

#### Database Management Prompts
- **`vertica_database_health_dashboard`** - Compact health overview
- **`vertica_database_system_monitor`** - Performance monitoring with alerting
- **`vertica_compact_health_report`** - Token-efficient status reporting

#### Query Optimization Prompts
- **`sql_query_safety_guard`** - Automated safety checks for large queries
- **`vertica_query_performance_analyzer`** - Deep-dive performance analysis with:
  - Execution plan parsing and cost analysis
  - Join strategy optimization recommendations
  - Projection health auditing
  - Concrete DDL suggestions for optimal projections
  - ROS container maintenance alerts
- **`vertica_sql_assistant`** - Expert SQL generation with Vertica-specific optimizations

### 🛡️ Enterprise Security Features

#### Multi-Level Permission System
- **Global Operation Controls**:
  - `ALLOW_INSERT_OPERATION` - Control INSERT permissions globally
  - `ALLOW_UPDATE_OPERATION` - Control UPDATE permissions globally  
  - `ALLOW_DELETE_OPERATION` - Control DELETE permissions globally
  - `ALLOW_DDL_OPERATION` - Control DDL permissions globally
- **Schema-Specific Permissions** - Fine-grained access control per schema:
  - `SCHEMA_INSERT_PERMISSIONS` - Per-schema INSERT control
  - `SCHEMA_UPDATE_PERMISSIONS` - Per-schema UPDATE control
  - `SCHEMA_DELETE_PERMISSIONS` - Per-schema DELETE control
  - `SCHEMA_DDL_PERMISSIONS` - Per-schema DDL control
- **Default Security Posture** - Read-only by default for production safety

#### Connection Security
- **SSL/TLS Support** with configurable certificate validation
- **Connection Pooling** with configurable limits (default: 10 connections)
- **Lazy Connection Initialization** for faster startup times
- **Connection Health Monitoring** with automatic recovery
- **Rate Limiting** (default: 60 requests/minute per client)

### 🚀 Multiple Transport Protocols

#### STDIO Transport (Default)
- **Local Integration** - Perfect for Claude Desktop and local development
- **Zero Network Configuration** - No ports or networking required
- **Process-per-Session** - Isolated execution contexts
- **Automatic Lifecycle Management** - Clean startup and shutdown

#### HTTP Transport (Streamable HTTP)
- **Remote Deployments** - Cloud and enterprise environments
- **RESTful API** - Standard HTTP/JSON communication
- **Configurable Endpoints** - Custom paths and ports
- **Stateless Sessions** - Scalable for load balancing
- **JSON Response Optimization** - Batch processing support

#### Server-Sent Events (SSE)
- **Real-Time Streaming** - Live data updates and notifications
- **Event-Driven Architecture** - Efficient for monitoring applications
- **Browser Compatible** - Direct browser integration support

### 🐳 Production-Ready Deployment

#### Docker Support
- **Multi-Transport Dockerfile** - Single image supporting all transports
- **Docker Compose Configuration** - Pre-configured services for all modes
- **Environment-Based Configuration** - 12-factor app compliance
- **Health Check Integration** - Container health monitoring
- **Auto-Restart Policies** - Production resilience

#### Configuration Management
- **Environment Variable Support** - `.env` file compatibility
- **CLI Parameter Override** - Flexible deployment options
- **Validation and Defaults** - Robust configuration handling
- **Multi-Environment Support** - Development, staging, production configs

#### Monitoring & Observability
- **Structured Logging** - Configurable verbosity levels (`-v`, `-vv`, `-vvv`)
- **Performance Metrics** - Query timing and resource usage
- **Error Tracking** - Comprehensive error handling and reporting
- **Cache Management** - Metadata caching with TTL controls

<!-- ### 🔧 Developer Experience

#### FastMCP Framework Integration
- Built on **FastMCP 2.11.3** - Latest production-ready framework
- **Automatic Type Generation** - Python type hints to MCP schema conversion
- **Decorator-Based API** - Clean, intuitive development patterns
- **Built-in Error Handling** - Automatic exception to MCP error conversion

#### Development Tools
- **uv Package Manager** - Fast, modern Python package management
- **Type Safety** - Full type annotations with mypy compatibility
- **Testing Framework** - Unit and integration test support
- **Hot Reload Support** - Development mode with automatic restarts -->

#### Client Integration Examples
- **Claude Desktop** - Complete setup instructions with configuration examples
- **VS Code** - MCP integration for GitHub Copilot
- **Cursor** - AI-powered code editor integration
- **Docker Configurations** - Multiple deployment patterns

### 🏗️ Architecture & Performance

#### Connection Management
- **Thread-Safe Connection Pool** - Concurrent request handling
- **Automatic Retry Logic** - Exponential backoff for failed connections
- **Connection Validation** - Health checks before query execution
- **Resource Cleanup** - Proper connection lifecycle management

#### Query Optimization
- **Result Size Management** - Automatic truncation for large results
- **Memory Monitoring** - Real-time memory usage tracking
- **Query Sanitization** - SQL injection prevention
- **Timeout Management** - Configurable query timeouts

#### Caching Strategy
- **Metadata Caching** - 5-minute TTL for schema information
- **Cache Invalidation** - Automatic cleanup of expired entries
- **Memory Efficiency** - LRU-based cache eviction

### 📋 System Requirements

#### Runtime Dependencies
- **Python** 3.11+ (Python 3.12 recommended)
- **Vertica Database** - Any supported version
- **Network Connectivity** - TCP access to Vertica port (default: 5433)

#### Python Dependencies
- **Core Framework**:
  - `mcp[cli]>=1.8.0` - Model Context Protocol implementation
  - `fastmcp>=2.11.3` - High-level MCP framework
- **Database Connectivity**:
  - `vertica-python>=1.4.0` - Official Vertica Python driver
- **Web Framework** (for HTTP/SSE transports):
  - `starlette>=0.46` - ASGI web framework
  - `uvicorn>=0.34` - ASGI server
- **Utilities**:
  - `click>=8.2.1` - CLI framework
  - `python-dotenv>=1.1.1` - Environment variable management

#### Development Dependencies
- **uv** - Modern Python package manager
- **Docker** (optional) - Container deployment
- **Git** - Version control

### 🔍 Testing & Validation

#### MCP Inspector Support
- **Protocol Validation** - MCP specification compliance testing
- **Tool Testing** - Interactive tool execution and validation
- **Transport Testing** - All transport protocols supported

#### Health Check Endpoints
- **Database Connectivity** - Automatic connection validation
- **Service Health** - Comprehensive health monitoring
- **Performance Metrics** - Real-time system status

### 📚 Documentation & Examples

#### Comprehensive Documentation
- **README.md** - Complete setup and usage guide
- **Environment Configuration** - `.env.example` with all options
- **Client Integration** - Step-by-step setup for popular clients
- **Docker Deployment** - Multiple deployment patterns
- **Security Best Practices** - Production deployment guidelines

#### Code Examples
- **Natural Language Queries** - Real-world usage examples
- **Performance Analysis** - Query optimization workflows
- **Health Monitoring** - System management examples
- **Complex Analytics** - Advanced use cases

### 🚨 Known Limitations

#### Current Restrictions
- **Read-Only Default** - Write operations disabled by default for safety
- **Single Database** - One database connection per server instance
- **Memory Limits** - Large result sets automatically truncated
- **STDIO Logging** - Must use stderr for STDIO transport compatibility

#### Future Enhancements
- Multi-database support
- Advanced authentication mechanisms
- Enhanced performance monitoring
- Custom projection recommendations

---

## Version History

| Version | Release Date | Key Features |
|---------|-------------|--------------|
| 0.1.0   | 2025-08-21  | Initial release with 11 tools, 5 prompts, enterprise security |

---

## Migration Guide

### From Direct Vertica Connections
If you're currently using direct Vertica Python connections, the Vertica MCP Server provides:
- **Standardized AI Interface** - Works with any MCP-compatible client
- **Enhanced Security** - Built-in permission controls and rate limiting
- **Better Performance** - Connection pooling and query optimization
- **Monitoring** - Real-time performance and health metrics

### Upgrading to Future Versions
This is the initial release. Future upgrade guides will be provided here as new versions are released.

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:
- Code style and standards
- Testing requirements
- Pull request process
- Issue reporting guidelines

---

## Support

- **GitHub Issues** - Bug reports and feature requests
- **Discussions** - Community support and Q&A
- **Documentation** - Comprehensive guides and examples

---

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.
