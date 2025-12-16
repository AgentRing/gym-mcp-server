"""
Gymnasium MCP Server

A Model Context Protocol (MCP) server that exposes any Gymnasium environment
as MCP tools, allowing agents to interact with Gym environments through
standard JSON interfaces.

Also provides an HTTP server with Swagger UI for REST API access.
"""

from .server import GymMCPServer
from .http_server import GymHTTPServer, create_app

__version__ = "0.3.0"
__all__ = ["GymMCPServer", "GymHTTPServer", "create_app"]
