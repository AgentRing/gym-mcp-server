"""
Tests for gym_mcp_server.server module.
"""

import pytest
from unittest.mock import Mock, patch
from gym_mcp_server.server import GymMCPServer, GymMCPAdapter


class TestGymMCPServer:
    """Test cases for GymMCPServer class."""

    def test_init_success(self, mock_gym_make, mock_env):
        """Test successful server initialization."""
        mock_gym_make.return_value = mock_env
        server = GymMCPServer("CartPole-v1", render_mode="ansi")

        assert server.env_id == "CartPole-v1"
        assert server.render_mode == "ansi"
        assert server.env == mock_env
        mock_gym_make.assert_called_once_with("CartPole-v1", render_mode="ansi")

    def test_init_failure(self, mock_gym_make):
        """Test server initialization failure."""
        mock_gym_make.side_effect = Exception("Environment not found")

        with pytest.raises(Exception, match="Environment not found"):
            GymMCPServer("InvalidEnv-v1")

    def test_reset_env_success(self, server_with_mock_env):
        """Test successful environment reset."""
        server, mock_env = server_with_mock_env
        result = server.reset_env()

        assert result["success"] is True
        assert result["done"] is False
        assert "observation" in result
        assert "info" in result
        mock_env.reset.assert_called_once_with()

    def test_reset_env_with_seed(self, server_with_mock_env):
        """Test environment reset with seed."""
        server, mock_env = server_with_mock_env
        result = server.reset_env(seed=42)

        assert result["success"] is True
        mock_env.reset.assert_called_once_with(seed=42)

    def test_reset_env_failure(self, server_with_mock_env):
        """Test environment reset failure."""
        server, mock_env = server_with_mock_env
        mock_env.reset.side_effect = Exception("Reset failed")

        result = server.reset_env()

        assert result["success"] is False
        assert result["done"] is True
        assert "error" in result
        assert result["observation"] is None

    def test_step_env_success(self, server_with_mock_env):
        """Test successful environment step."""
        server, mock_env = server_with_mock_env
        result = server.step_env(0)

        assert result["success"] is True
        assert result["reward"] == 1.0
        assert result["done"] is False
        assert result["truncated"] is False
        assert "observation" in result
        assert "info" in result
        mock_env.step.assert_called_once_with(0)

    def test_step_env_with_list_action(self, server_with_mock_env):
        """Test environment step with list action (single element)."""
        server, mock_env = server_with_mock_env
        result = server.step_env([0])

        assert result["success"] is True
        mock_env.step.assert_called_once_with(0)

    def test_step_env_failure(self, server_with_mock_env):
        """Test environment step failure."""
        server, mock_env = server_with_mock_env
        mock_env.step.side_effect = Exception("Step failed")

        result = server.step_env(0)

        assert result["success"] is False
        assert result["reward"] == 0.0
        assert result["done"] is True
        assert "error" in result

    def test_render_env_success(self, server_with_mock_env):
        """Test successful environment render."""
        server, mock_env = server_with_mock_env
        result = server.render_env()

        assert result["success"] is True
        assert "render" in result
        assert "mode" in result
        mock_env.render.assert_called_once()

    def test_render_env_with_mode(self, server_with_mock_env):
        """Test environment render with specific mode."""
        server, mock_env = server_with_mock_env
        result = server.render_env("rgb_array")

        assert result["success"] is True
        mock_env.render.assert_called_once()

    def test_render_env_failure(self, server_with_mock_env):
        """Test environment render failure."""
        server, mock_env = server_with_mock_env
        mock_env.render.side_effect = Exception("Render failed")

        result = server.render_env()

        assert result["success"] is False
        assert "error" in result
        assert result["render"] is None

    def test_close_env_success(self, server_with_mock_env):
        """Test successful environment close."""
        server, mock_env = server_with_mock_env
        result = server.close_env()

        assert result["success"] is True
        assert result["status"] == "closed"
        mock_env.close.assert_called_once()

    def test_close_env_failure(self, server_with_mock_env):
        """Test environment close failure."""
        server, mock_env = server_with_mock_env
        mock_env.close.side_effect = Exception("Close failed")

        result = server.close_env()

        assert result["success"] is False
        assert result["status"] == "error"
        assert "error" in result

    def test_get_env_info_success(self, server_with_mock_env):
        """Test successful environment info retrieval."""
        server, mock_env = server_with_mock_env
        result = server.get_env_info()

        assert result["success"] is True
        assert "env_info" in result
        assert result["env_info"]["id"] == "CartPole-v1"

    def test_get_env_info_failure(self, server_with_mock_env):
        """Test environment info retrieval failure."""
        server, mock_env = server_with_mock_env
        # Mock the get_environment_info function to raise an exception
        with patch("gym_mcp_server.server.get_environment_info") as mock_get_info:
            mock_get_info.side_effect = Exception("Info failed")
            result = server.get_env_info()

            assert result["success"] is False
            assert "error" in result

    def test_get_available_tools(self, server_with_mock_env):
        """Test getting available tools."""
        server, mock_env = server_with_mock_env
        result = server.get_available_tools()

        assert result["success"] is True
        assert "tools" in result
        assert "reset_env" in result["tools"]
        assert "step_env" in result["tools"]
        assert "render_env" in result["tools"]
        assert "close_env" in result["tools"]
        assert "get_env_info" in result["tools"]


class TestGymMCPAdapter:
    """Test cases for GymMCPAdapter class."""

    def test_init(self, mock_gym_make, mock_env):
        """Test adapter initialization."""
        mock_gym_make.return_value = mock_env
        adapter = GymMCPAdapter("CartPole-v1", render_mode="ansi")

        assert adapter.env_id == "CartPole-v1"
        assert adapter.server is not None
        assert isinstance(adapter.server, GymMCPServer)

    def test_call_tool_reset_env(self, adapter_with_mock_env):
        """Test calling reset_env tool."""
        adapter, mock_env = adapter_with_mock_env
        result = adapter.call_tool("reset_env", seed=42)

        assert result["success"] is True
        mock_env.reset.assert_called_once_with(seed=42)

    def test_call_tool_step_env(self, adapter_with_mock_env):
        """Test calling step_env tool."""
        adapter, mock_env = adapter_with_mock_env
        result = adapter.call_tool("step_env", action=0)

        assert result["success"] is True
        mock_env.step.assert_called_once_with(0)

    def test_call_tool_render_env(self, adapter_with_mock_env):
        """Test calling render_env tool."""
        adapter, mock_env = adapter_with_mock_env
        result = adapter.call_tool("render_env", mode="ansi")

        assert result["success"] is True
        mock_env.render.assert_called_once()

    def test_call_tool_close_env(self, adapter_with_mock_env):
        """Test calling close_env tool."""
        adapter, mock_env = adapter_with_mock_env
        result = adapter.call_tool("close_env")

        assert result["success"] is True
        mock_env.close.assert_called_once()

    def test_call_tool_get_env_info(self, adapter_with_mock_env):
        """Test calling get_env_info tool."""
        adapter, mock_env = adapter_with_mock_env
        result = adapter.call_tool("get_env_info")

        assert result["success"] is True
        assert "env_info" in result

    def test_call_tool_get_available_tools(self, adapter_with_mock_env):
        """Test calling get_available_tools tool."""
        adapter, mock_env = adapter_with_mock_env
        result = adapter.call_tool("get_available_tools")

        assert result["success"] is True
        assert "tools" in result

    def test_call_tool_unknown(self, adapter_with_mock_env):
        """Test calling unknown tool."""
        adapter, mock_env = adapter_with_mock_env
        result = adapter.call_tool("unknown_tool")

        assert result["success"] is False
        assert "error" in result
        assert "Unknown tool" in result["error"]

    @patch("builtins.input")
    @patch("builtins.print")
    def test_run_interactive_quit(self, mock_print, mock_input, adapter_with_mock_env):
        """Test interactive mode quit command."""
        adapter, mock_env = adapter_with_mock_env
        mock_input.side_effect = ["quit"]

        adapter.run_interactive()

        # Should print help and then exit
        assert mock_print.call_count > 0

    @patch("builtins.input")
    @patch("builtins.print")
    def test_run_interactive_reset(self, mock_print, mock_input, adapter_with_mock_env):
        """Test interactive mode reset command."""
        adapter, mock_env = adapter_with_mock_env
        mock_input.side_effect = ["reset", "quit"]

        adapter.run_interactive()

        mock_env.reset.assert_called_once()

    @patch("builtins.input")
    @patch("builtins.print")
    def test_run_interactive_step(self, mock_print, mock_input, adapter_with_mock_env):
        """Test interactive mode step command."""
        adapter, mock_env = adapter_with_mock_env
        mock_input.side_effect = ["step 0", "quit"]

        adapter.run_interactive()

        mock_env.step.assert_called_once_with(0)

    @patch("builtins.input")
    @patch("builtins.print")
    def test_run_interactive_render(
        self, mock_print, mock_input, adapter_with_mock_env
    ):
        """Test interactive mode render command."""
        adapter, mock_env = adapter_with_mock_env
        mock_input.side_effect = ["render", "quit"]

        adapter.run_interactive()

        mock_env.render.assert_called_once()

    @patch("builtins.input")
    @patch("builtins.print")
    def test_run_interactive_info(self, mock_print, mock_input, adapter_with_mock_env):
        """Test interactive mode info command."""
        adapter, mock_env = adapter_with_mock_env
        mock_input.side_effect = ["info", "quit"]

        adapter.run_interactive()

        # Info command should be called (tested through the server method)

    @patch("builtins.input")
    @patch("builtins.print")
    def test_run_interactive_close(self, mock_print, mock_input, adapter_with_mock_env):
        """Test interactive mode close command."""
        adapter, mock_env = adapter_with_mock_env
        mock_input.side_effect = ["close", "quit"]

        adapter.run_interactive()

        mock_env.close.assert_called_once()

    @patch("builtins.input")
    @patch("builtins.print")
    def test_run_interactive_unknown_command(
        self, mock_print, mock_input, adapter_with_mock_env
    ):
        """Test interactive mode unknown command."""
        adapter, mock_env = adapter_with_mock_env
        mock_input.side_effect = ["unknown", "quit"]

        adapter.run_interactive()

        # Should handle unknown command gracefully

    @patch("builtins.input")
    @patch("builtins.print")
    def test_run_interactive_keyboard_interrupt(
        self, mock_print, mock_input, adapter_with_mock_env
    ):
        """Test interactive mode keyboard interrupt."""
        adapter, mock_env = adapter_with_mock_env
        mock_input.side_effect = KeyboardInterrupt()

        adapter.run_interactive()

        # Should handle KeyboardInterrupt gracefully


class TestMainFunction:
    """Test cases for main function."""

    @patch("gym_mcp_server.server.GymMCPAdapter")
    @patch("gym_mcp_server.server.argparse.ArgumentParser")
    def test_main_success(self, mock_parser_class, mock_adapter_class):
        """Test successful main function execution."""
        # Mock argument parser
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        mock_args = Mock()
        mock_args.env = "CartPole-v1"
        mock_args.render_mode = "ansi"
        mock_args.interactive = False
        mock_args.host = "localhost"
        mock_args.port = 8000
        mock_parser.parse_args.return_value = mock_args

        # Mock adapter
        mock_adapter = Mock()
        mock_adapter_class.return_value = mock_adapter

        from gym_mcp_server.server import main

        result = main()

        assert result == 0
        mock_adapter_class.assert_called_once_with("CartPole-v1", "ansi")

    @patch("gym_mcp_server.server.GymMCPAdapter")
    @patch("gym_mcp_server.server.argparse.ArgumentParser")
    def test_main_interactive(self, mock_parser_class, mock_adapter_class):
        """Test main function with interactive mode."""
        # Mock argument parser
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        mock_args = Mock()
        mock_args.env = "CartPole-v1"
        mock_args.render_mode = "ansi"
        mock_args.interactive = True
        mock_args.host = "localhost"
        mock_args.port = 8000
        mock_parser.parse_args.return_value = mock_args

        # Mock adapter
        mock_adapter = Mock()
        mock_adapter_class.return_value = mock_adapter

        from gym_mcp_server.server import main

        result = main()

        assert result == 0
        mock_adapter.run_interactive.assert_called_once()

    @patch("gym_mcp_server.server.GymMCPAdapter")
    @patch("gym_mcp_server.server.argparse.ArgumentParser")
    def test_main_failure(self, mock_parser_class, mock_adapter_class):
        """Test main function failure."""
        # Mock argument parser
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        mock_args = Mock()
        mock_args.env = "CartPole-v1"
        mock_args.render_mode = "ansi"
        mock_args.interactive = False
        mock_args.host = "localhost"
        mock_args.port = 8000
        mock_parser.parse_args.return_value = mock_args

        # Mock adapter to raise exception
        mock_adapter_class.side_effect = Exception("Adapter failed")

        from gym_mcp_server.server import main

        result = main()

        assert result == 1
