"""
Gymnasium MCP Server

A Model Context Protocol (MCP) server that exposes any Gymnasium environment
as MCP tools, allowing agents to interact with Gym environments through
standard JSON interfaces.

Also provides HTTP/REST and gRPC servers for different use cases.
"""

from .servers import (
    BaseGymServer,
    GymMCPServer,
    GymHTTPServer,
    GymGRPCServer,
    create_app,
)
from .service import GymService

__version__ = "0.3.0"
__all__ = [
    "BaseGymServer",
    "GymMCPServer",
    "GymHTTPServer",
    "GymGRPCServer",
    "GymService",
    "create_app",
]
