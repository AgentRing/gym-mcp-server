"""
Gymnasium MCP Server

A Model Context Protocol (MCP) server that exposes any Gymnasium environment
as MCP tools, allowing agents to interact with Gym environments through
standard JSON interfaces.

Also provides an HTTP/REST server (with Swagger UI) for different use cases.
"""

from .servers import (
    GymHTTPServer,
    create_app,
)
from .service import GymService
from .server import GymMCPServer

__version__ = "0.3.0"
__all__ = [
    "GymHTTPServer",
    "GymService",
    "GymMCPServer",
    "create_app",
]
