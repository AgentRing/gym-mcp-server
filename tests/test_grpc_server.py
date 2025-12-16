"""
Tests specific to gRPC server implementation.

Note: Common server tests are in test_servers_unified.py and run
against all three server types (MCP, HTTP, gRPC).
"""

import pytest
from unittest.mock import patch, MagicMock


class TestGymGRPCServerSpecific:
    """Test cases specific to gRPC server (not shared with other servers)."""

    def test_init_without_proto(self, mock_env):
        """Test that initialization fails without proto modules."""
        with patch("gym_mcp_server.servers.grpc.gym_service_pb2", None), \
             patch("gym_mcp_server.servers.grpc.gym_service_pb2_grpc", None), \
             patch("gymnasium.make", return_value=mock_env):
            from gym_mcp_server.servers.grpc import GymGRPCServer
            
            with pytest.raises(ImportError, match="gRPC proto code not generated"):
                GymGRPCServer(env_id="CartPole-v1")

    @pytest.mark.parametrize("server_type", ["grpc"], indirect=True)
    def test_grpc_server_stop_behavior(self, server_instance):
        """Test gRPC-specific stop behavior."""
        server, _, _ = server_instance
        
        # Create a mock gRPC server
        mock_grpc_server = MagicMock()
        server._server = mock_grpc_server
        
        server.stop(grace=5)
        
        assert mock_grpc_server.stop.called
        # Check that grace period was passed
        call_kwargs = mock_grpc_server.stop.call_args
        assert call_kwargs[1].get('grace') == 5 or call_kwargs[0][0] == 5
        assert server._server is None

    @pytest.mark.parametrize("server_type", ["grpc"], indirect=True)
    def test_grpc_server_thread_lifecycle(self, server_instance):
        """Test gRPC server thread lifecycle."""
        server, _, _ = server_instance
        
        # Initially not alive
        assert not server.is_alive()
        
        # Create a mock thread
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        server._thread = mock_thread
        
        assert server.is_alive()
        
        # Thread is None
        server._thread = None
        assert not server.is_alive()


class TestGymServiceServicer:
    """Test cases for GymServiceServicer class (gRPC-specific)."""

    def test_servicer_initialization(self, mock_env, mock_proto_modules):
        """Test servicer initialization."""
        mock_pb2, mock_pb2_grpc = mock_proto_modules
        
        with patch("gym_mcp_server.servers.grpc.gym_service_pb2", mock_pb2), \
             patch("gym_mcp_server.servers.grpc.gym_service_pb2_grpc", mock_pb2_grpc), \
             patch("gymnasium.make", return_value=mock_env):
            from gym_mcp_server.servers.grpc import GymGRPCServer
            
            server = GymGRPCServer(env_id="CartPole-v1")
            
            # The servicer is created internally, but we can test the server uses service
            assert server._service is not None
            assert server._service.env_id == "CartPole-v1"
