#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path

def get_claude_config_path():
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", "")) / "Claude" / "claude_desktop_config.json"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    else:
        # Linux is not officially supported by Claude Desktop yet, but just in case
        return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"

def get_cursor_config_path():
    if sys.platform == "win32":
        return Path(os.environ.get("USERPROFILE", "")) / ".cursor" / "mcp.json"
    else:
        return Path.home() / ".cursor" / "mcp.json"

def update_json_config(config_path, mcp_key, server_config):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except json.JSONDecodeError:
            config = {"mcpServers": {}}
    else:
        config = {"mcpServers": {}}
        
    if "mcpServers" not in config:
        config["mcpServers"] = {}
        
    config["mcpServers"][mcp_key] = server_config
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"✅ Updated {config_path}")

def main():
    print("="*50)
    print("🚀 Vertica MCP - 1-Click Client Setup")
    print("="*50)
    
    # 1. Setup .env
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        print("\n📝 Creating .env file...")
        host = input("Vertica Host (e.g. localhost): ") or "localhost"
        db = input("Database Name: ")
        user = input("Username: ")
        pwd = input("Password: ")
        
        with open(env_path, "w") as f:
            f.write(f"VERTICA_HOST={host}\n")
            f.write("VERTICA_PORT=5433\n")
            f.write(f"VERTICA_DATABASE={db}\n")
            f.write(f"VERTICA_USER={user}\n")
            f.write(f"VERTICA_PASSWORD={pwd}\n")
            f.write("VERTICA_AUTH_MODE=basic\n")
        print("✅ Created .env file")
    else:
        print("\n✅ Found existing .env file")
        
    env_abs_path = str(env_path.absolute())
    
    print("\n📦 How did you install Vertica MCP?")
    print("1. From source (using 'uv') [Default]")
    print("2. From PyPI (using 'pip install vertica-mcp')")
    install_method = input("Select an option [1/2]: ").strip()
    
    if install_method == "2":
        server_config = {
            "command": "vertica-mcp",
            "args": [
                "--transport", "stdio",
                "--env-file", env_abs_path
            ]
        }
    else:
        # Default to uv
        uv_path = "uv" # assumes uv is in PATH
        server_config = {
            "command": uv_path,
            "args": [
                "run",
                "vertica-mcp",
                "--transport", "stdio",
                "--env-file", env_abs_path
            ]
        }
    
    print("\n🔌 Configuring AI Clients...")
    
    # Claude Desktop
    claude_path = get_claude_config_path()
    try:
        update_json_config(claude_path, "vertica-mcp", server_config)
        print("   -> Claude Desktop configured! Please restart Claude Desktop.")
    except Exception as e:
        print(f"   ❌ Failed to configure Claude: {e}")

    # Cursor
    cursor_path = get_cursor_config_path()
    try:
        update_json_config(cursor_path, "vertica-mcp", server_config)
        print("   -> Cursor configured! Please restart Cursor.")
    except Exception as e:
        print(f"   ❌ Failed to configure Cursor: {e}")

    print("\n🎉 All done! Your AI clients are now connected to Vertica.")
    print("="*50)

if __name__ == "__main__":
    main()
