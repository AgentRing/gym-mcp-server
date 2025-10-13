"""
Tests for gym_mcp_server.server module.
"""

from unittest.mock import Mock, patch


class TestMainFunction:
    """Test cases for main function."""

    @patch("gym_mcp_server.__main__.GymMCPServer")
    @patch("gym_mcp_server.__main__.argparse.ArgumentParser")
    def test_main_success(self, mock_parser_class, mock_gym_server_class):
        """Test successful main function execution (MCP server mode)."""
        # Mock argument parser
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        mock_args = Mock()
        mock_args.env = "CartPole-v1"
        mock_args.render_mode = "rgb_array"
        mock_args.transport = "stdio"
        mock_args.host = "localhost"
        mock_args.port = 8000
        mock_parser.parse_args.return_value = mock_args

        # Mock the GymMCPServer
        mock_gym_server = Mock()
        mock_gym_server_class.return_value = mock_gym_server

        from gym_mcp_server.__main__ import main

        result = main()

        assert result == 0
        # Verify the server was created and run was called
        assert mock_gym_server_class.called
        assert mock_gym_server.run.called

    @patch("gym_mcp_server.__main__.GymMCPServer")
    @patch("gym_mcp_server.__main__.argparse.ArgumentParser")
    def test_main_failure(self, mock_parser_class, mock_gym_server_class):
        """Test main function failure."""
        # Mock argument parser
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        mock_args = Mock()
        mock_args.env = "CartPole-v1"
        mock_args.render_mode = "ansi"
        mock_args.transport = "stdio"
        mock_args.host = "localhost"
        mock_args.port = 8000
        mock_parser.parse_args.return_value = mock_args

        # Mock GymMCPServer to raise exception
        mock_gym_server_class.side_effect = Exception("Server creation failed")

        from gym_mcp_server.__main__ import main

        result = main()

        assert result == 1
