# setup.py
from setuptools import setup, find_packages

setup(
    name="vertica-mcp",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "vertica-mcp=vertica_mcp.cli:cli",
        ],
    },
    install_requires=[
        "click>=8.2.1",
        "mcp[cli]>=1.8.0",
        "python-dotenv>=1.1.1",
        "vertica-python>=1.4.0",
        "setuptools>=61.0",
        "starlette>=0.46",
        "uvicorn>=0.34",
        "fastmcp>=2.11.3"
    ],
)
