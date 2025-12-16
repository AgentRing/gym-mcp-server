"""
Tests for gym_mcp_server.__init__ module.
"""

from gym_mcp_server import (
    BaseGymServer,
    GymMCPServer,
    GymHTTPServer,
    GymGRPCServer,
    GymService,
    create_app,
    __version__,
)


class TestInitModule:
    """Test cases for __init__ module."""

    def test_version(self):
        """Test that __version__ is defined."""
        assert __version__ == "0.3.0"

    def test_imports(self):
        """Test that main classes can be imported."""
        assert GymMCPServer is not None
        assert GymHTTPServer is not None
        assert GymGRPCServer is not None
        assert BaseGymServer is not None
        assert GymService is not None
        assert create_app is not None

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        import gym_mcp_server

        assert hasattr(gym_mcp_server, "__all__")
        assert "GymMCPServer" in gym_mcp_server.__all__
        assert "GymHTTPServer" in gym_mcp_server.__all__
        assert "GymGRPCServer" in gym_mcp_server.__all__
        assert "BaseGymServer" in gym_mcp_server.__all__
        assert "GymService" in gym_mcp_server.__all__
        assert "create_app" in gym_mcp_server.__all__

    def test_module_docstring(self):
        """Test that module has proper docstring."""
        import gym_mcp_server

        assert gym_mcp_server.__doc__ is not None
        assert "Gymnasium MCP Server" in gym_mcp_server.__doc__
        assert "Model Context Protocol" in gym_mcp_server.__doc__
