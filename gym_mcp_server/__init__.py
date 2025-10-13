"""
Gymnasium MCP Server

A Model Context Protocol (MCP) server that exposes any Gymnasium environment
as MCP tools, allowing agents to interact with Gym environments through
standard JSON interfaces.
"""

from .server import GymMCPServer

__version__ = "0.1.0"
__all__ = ["GymMCPServer"]
