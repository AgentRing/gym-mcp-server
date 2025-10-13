"""
Tests for gym_mcp_server.__init__ module.
"""

from gym_mcp_server import GymMCPServer, __version__


class TestInitModule:
    """Test cases for __init__ module."""

    def test_version(self):
        """Test that __version__ is defined."""
        assert __version__ == "0.1.0"

    def test_imports(self):
        """Test that main classes can be imported."""
        assert GymMCPServer is not None

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        import gym_mcp_server

        assert hasattr(gym_mcp_server, "__all__")
        assert "GymMCPServer" in gym_mcp_server.__all__
        assert len(gym_mcp_server.__all__) == 1

    def test_module_docstring(self):
        """Test that module has proper docstring."""
        import gym_mcp_server

        assert gym_mcp_server.__doc__ is not None
        assert "Gymnasium MCP Server" in gym_mcp_server.__doc__
        assert "Model Context Protocol" in gym_mcp_server.__doc__
