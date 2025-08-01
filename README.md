# Vertica MCP

This repository contains resources and scripts for managing Vertica Multi-Cluster Processing (MCP) environments.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Commands](#commands)
- [Contributing](#contributing)
- [License](#license)

## Overview

Vertica MCP enables scalable, distributed analytics using Vertica clusters. This repo provides scripts and documentation to help set up, manage, and monitor MCP clusters.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/vertica-mcp.git
    cd vertica-mcp
    ```

2. **Claude Desktop Configuration:**
    Create or update your `claude_desktop_config.json` with the following template:
    ```json
    {
      "mcpServers": {
        "vertica-mcp": {
          "command": "<path_to_uv.exe>",
          "args": [
            "run",
            "--with",
            "mcp[cli]",
            "--with",
            "pydantic",
            "--with",
            "starlette",
            "--with",
            "uvicorn",
            "--with",
            "vertica-python",
            "mcp",
            "run",
            "<path_to_vertica_mcp>/vertica_mcp/server.py"
          ],
          "env": {
            "PYTHONPATH": "<path_to_environement>",
            "VERTICA_HOST": "<your_vertica_host>",
            "VERTICA_PORT": "5433",
            "VERTICA_DATABASE": "<your_database_name>",
            "VERTICA_USER": "<your_username>",
            "VERTICA_PASSWORD": "<your_password>",
            "VERTICA_CONNECTION_LIMIT": "10",
            "VERTICA_SSL": "false",
            "VERTICA_SSL_REJECT_UNAUTHORIZED": "true"
          }
        }
      }
    }
    ```
    Replace the placeholders:
    - `<path_to_uv.exe>`: Full path to your uv.exe installation
    - `<path_to_environement>`: Full path to your vertica-mcp project environement.
    - `<your_vertica_host>`: Your Vertica database host
    - `<your_database_name>`: Your Vertica database name
    - `<your_username>`: Your Vertica username
    - `<your_password>`: Your Vertica password

## Usage
### Using MCP Inspector (Windows)

To use MCP Inspector, set your `PYTHONPATH` to the current directory and run the server:

```pwsh
$env:PYTHONPATH = (Resolve-Path .).Path
mcp dev .\vertica_mcp\server.py
```

This will start the MCP server in development mode with the inspector enabled.

### Running the Server with UV (Windows)

**Command:**

```pwsh
uv
```

**Arguments:**

```pwsh
run --with mcp --with pydantic --with uvicorn --with starlette --with vertica-python mcp run vertica_mcp/server.py
```

This will:
- Create an isolated environment using `uv`
- Install MCP and all required dependencies
- Run the server in development mode
- Auto-reload when code changes
- Configure the environment from `.env` or environment variables

### Running the Server

You can start the server in two ways:

1. **Using UV (Recommended for Claude Desktop):**

    ```bash
    uv run mcp install ./vertica_mcp/server.py
    ```

    This command:
    - Creates an isolated environment using `uv`
    - Installs MCP and all required dependencies
    - Runs the server in development mode
    - Auto-reloads when code changes
    - Configures the environment from `.env` or environment variables

2. **Using Claude Desktop Configuration:**
  Edit `claude_desktop_config.json` as shown in the installation section.

## Contributing

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/foo`)
3. Commit your changes (`git commit -am 'Add feature'`)
4. Push to the branch (`git push origin feature/foo`)
5. Create a pull request

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.


