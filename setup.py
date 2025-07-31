# setup.py
from setuptools import setup, find_packages

setup(
    name="vertica-mcp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "vertica-python",
        "pydantic",
        "starlette",
        "uvicorn",
        "mcp",
    ],
)