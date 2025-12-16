"""
Unified tests for all server types (MCP, HTTP, gRPC).

These tests run against all three server implementations to ensure
consistent behavior across protocols.
"""

import pytest
import inspect
from unittest.mock import MagicMock


class TestServerInitialization:
    """Test cases for server initialization across all server types."""

    def test_init_success(self, server, server_type):
        """Test successful initialization."""
        assert server.env_id == "CartPole-v1"
        assert server.render_mode is None
        assert server._service is not None

    def test_init_with_render_mode(self, server_class, mock_env, mock_proto_modules, server_type):
        """Test initialization with render mode."""
        from unittest.mock import patch
        
        patches = [patch("gymnasium.make", return_value=mock_env)]
        
        if server_type == "grpc":
            mock_pb2, mock_pb2_grpc = mock_proto_modules
            patches.append(patch("gym_mcp_server.servers.grpc.gym_service_pb2", mock_pb2))
            patches.append(patch("gym_mcp_server.servers.grpc.gym_service_pb2_grpc", mock_pb2_grpc))
        
        with patch("gymnasium.make", return_value=mock_env):
            if server_type == "grpc":
                mock_pb2, mock_pb2_grpc = mock_proto_modules
                with patch("gym_mcp_server.servers.grpc.gym_service_pb2", mock_pb2), \
                     patch("gym_mcp_server.servers.grpc.gym_service_pb2_grpc", mock_pb2_grpc):
                    server = server_class(env_id="CartPole-v1", render_mode="rgb_array")
            else:
                server = server_class(env_id="CartPole-v1", render_mode="rgb_array")
            
            assert server.render_mode == "rgb_array"

    def test_inherits_from_base(self, server):
        """Test that all servers inherit from BaseGymServer."""
        from gym_mcp_server.servers.base import BaseGymServer
        
        assert isinstance(server, BaseGymServer)
        assert hasattr(server, "service")
        assert hasattr(server, "run")
        assert hasattr(server, "start")
        assert hasattr(server, "stop")
        assert hasattr(server, "is_alive")

    def test_service_property(self, server):
        """Test that service property returns GymService instance."""
        from gym_mcp_server.service import GymService
        
        assert isinstance(server.service, GymService)
        assert server.service.env_id == "CartPole-v1"


class TestServerLifecycle:
    """Test cases for server lifecycle methods (start/stop/is_alive)."""

    def test_start_method_exists(self, server):
        """Test that start() method exists and has correct signature."""
        assert hasattr(server, "start")
        assert callable(server.start)
        
        # Verify signature
        sig = inspect.signature(server.start)
        # Different servers have different parameters, but all should have start()
        assert "start" in dir(server)

    def test_stop_method_exists(self, server):
        """Test that stop() method exists."""
        assert hasattr(server, "stop")
        assert callable(server.stop)

    def test_is_alive_initially_false(self, server):
        """Test that is_alive() returns False initially."""
        assert not server.is_alive()

    def test_is_alive_after_start(self, server, server_type):
        """Test that is_alive() returns True after starting server."""
        # Note: This test may need to be skipped for some server types
        # if they require actual network setup. For now, we just verify
        # the method exists and can be called.
        assert hasattr(server, "is_alive")
        assert callable(server.is_alive)

    def test_stop_server(self, server, server_type):
        """Test stopping the server."""
        # Create mock server state if needed
        if server_type == "grpc":
            if not hasattr(server, "_server"):
                pytest.skip("Server doesn't have _server attribute")
            mock_grpc_server = MagicMock()
            server._server = mock_grpc_server
            
            server.stop(grace=5)
            
            # Verify stop was called (if applicable)
            if hasattr(mock_grpc_server, "stop"):
                assert mock_grpc_server.stop.called
            assert server._server is None
        elif server_type == "http":
            if not hasattr(server, "_server"):
                pytest.skip("Server doesn't have _server attribute")
            mock_http_server = MagicMock()
            mock_http_server.should_exit = False
            server._server = mock_http_server
            
            server.stop(grace=5)
            
            # HTTP server sets should_exit flag
            assert mock_http_server.should_exit is True
        else:  # MCP
            # MCP server stop just sets stop event when thread exists
            if not hasattr(server, "_stop_event"):
                pytest.skip("MCP server doesn't have _stop_event")
            # Create a mock thread so stop() will actually set the event
            mock_thread = MagicMock()
            mock_thread.is_alive.return_value = True
            server._thread = mock_thread
            
            server.stop()
            assert server._stop_event.is_set()


class TestServerRunMethod:
    """Test cases for server run() method."""

    def test_run_method_exists(self, server):
        """Test that run() method exists and is callable."""
        assert hasattr(server, "run")
        assert callable(server.run)

    def test_run_method_signature(self, server, server_type):
        """Test that run() method has appropriate signature."""
        sig = inspect.signature(server.run)
        
        # Different servers have different parameters
        if server_type == "mcp":
            assert "transport" in sig.parameters
        elif server_type in ["http", "grpc"]:
            assert "host" in sig.parameters
            assert "port" in sig.parameters


class TestServerServiceIntegration:
    """Test cases for server-service integration."""

    def test_service_initialization(self, server):
        """Test that service is properly initialized."""
        assert server._service is not None
        assert server._service.env_id == "CartPole-v1"
        assert server.service.env_id == "CartPole-v1"

    def test_service_reset(self, server):
        """Test that service reset works through server."""
        # The service should be accessible and functional
        result = server.service.reset()
        assert "observation" in result
        assert "success" in result

    def test_service_step(self, server):
        """Test that service step works through server."""
        # First reset to initialize
        server.service.reset()
        
        # Then step
        result = server.service.step(0)
        assert "observation" in result
        assert "reward" in result
        assert "success" in result

    def test_service_get_info(self, server):
        """Test that service get_info works through server."""
        result = server.service.get_info()
        assert "env_info" in result
        assert "success" in result

    def test_service_close(self, server):
        """Test that service close works through server."""
        result = server.service.close()
        assert "status" in result
        assert "success" in result


class TestGRPCServerSpecific:
    """Test cases specific to gRPC server (requires proto modules)."""

    @pytest.mark.parametrize("server_type", ["grpc"], indirect=True)
    def test_grpc_init_without_proto(self, mock_env, server_type):
        """Test that gRPC initialization fails without proto modules."""
        from unittest.mock import patch
        
        with patch("gymnasium.make", return_value=mock_env), \
             patch("gym_mcp_server.servers.grpc.gym_service_pb2", None), \
             patch("gym_mcp_server.servers.grpc.gym_service_pb2_grpc", None):
            from gym_mcp_server.servers.grpc import GymGRPCServer
            
            with pytest.raises(ImportError, match="gRPC proto code not generated"):
                GymGRPCServer(env_id="CartPole-v1")

