"""
Pytest configuration and fixtures for gym-mcp-server tests.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Type, Any


@pytest.fixture
def mock_env():
    """Create a mock Gymnasium environment for testing."""
    env = Mock()
    env.spec = Mock()
    env.spec.id = "CartPole-v1"
    env.action_space = Mock()
    env.action_space.n = 2
    env.action_space.shape = (1,)
    env.observation_space = Mock()
    env.observation_space.shape = (4,)
    env.reward_range = (-1.0, 1.0)
    env.close = Mock()
    env.reset = Mock(return_value=(np.array([0.1, 0.2, 0.3, 0.4]), {}))
    env.step = Mock(
        return_value=(np.array([0.1, 0.2, 0.3, 0.4]), 1.0, False, False, {})
    )
    env.render = Mock(return_value="rendered output")
    return env


@pytest.fixture
def mock_gym_make():
    """Mock gym.make function."""
    with patch("gymnasium.make") as mock_make:
        yield mock_make


@pytest.fixture
def mock_proto_modules():
    """Mock the proto modules for testing gRPC server."""
    # Create a mock module structure that mimics the proto generated code
    mock_pb2 = MagicMock()
    mock_pb2_grpc = MagicMock()
    
    # Create mock message classes that can be instantiated
    class MockMessage:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        
        def HasField(self, field):
            return hasattr(self, field) and getattr(self, field) is not None
    
    class MockObservation(MockMessage):
        def CopyFrom(self, other):
            pass
    
    class MockScalarObservation(MockMessage):
        pass
    
    class MockArrayObservation(MockMessage):
        pass
    
    class MockDictObservation(MockMessage):
        def __init__(self):
            self.values = {}
    
    class MockAction(MockMessage):
        pass
    
    class MockResetRequest(MockMessage):
        pass
    
    class MockResetResponse(MockMessage):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.info = {}
            self.observation = MockObservation()
    
    class MockStepRequest(MockMessage):
        pass
    
    class MockStepResponse(MockMessage):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.info = {}
            self.observation = MockObservation()
    
    class MockRenderRequest(MockMessage):
        pass
    
    class MockRenderResponse(MockMessage):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.shape = []
    
    class MockGetInfoRequest(MockMessage):
        pass
    
    class MockGetInfoResponse(MockMessage):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.env_info = MockMessage()
    
    class MockCloseRequest(MockMessage):
        pass
    
    class MockCloseResponse(MockMessage):
        pass
    
    class MockHealthRequest(MockMessage):
        pass
    
    class MockHealthResponse(MockMessage):
        pass
    
    class MockEnvInfo(MockMessage):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.reward_range = MockMessage()
            self.action_space_shape = []
            self.observation_space_shape = []
            self.observation_space_spaces = {}
    
    # Assign to mock module
    mock_pb2.ResetRequest = MockResetRequest
    mock_pb2.ResetResponse = MockResetResponse
    mock_pb2.StepRequest = MockStepRequest
    mock_pb2.StepResponse = MockStepResponse
    mock_pb2.RenderRequest = MockRenderRequest
    mock_pb2.RenderResponse = MockRenderResponse
    mock_pb2.GetInfoRequest = MockGetInfoRequest
    mock_pb2.GetInfoResponse = MockGetInfoResponse
    mock_pb2.CloseRequest = MockCloseRequest
    mock_pb2.CloseResponse = MockCloseResponse
    mock_pb2.HealthRequest = MockHealthRequest
    mock_pb2.HealthResponse = MockHealthResponse
    mock_pb2.Observation = MockObservation
    mock_pb2.ScalarObservation = MockScalarObservation
    mock_pb2.ArrayObservation = MockArrayObservation
    mock_pb2.DictObservation = MockDictObservation
    mock_pb2.Action = MockAction
    mock_pb2.EnvInfo = MockEnvInfo
    
    # Create servicer base class
    class MockServicerBase:
        pass
    
    mock_pb2_grpc.GymServiceServicer = MockServicerBase
    mock_pb2_grpc.add_GymServiceServicer_to_server = Mock()
    
    return mock_pb2, mock_pb2_grpc


# Parametrize server types for unified testing
@pytest.fixture(params=["mcp", "http", "grpc"])
def server_type(request):
    """Parametrized fixture for server types."""
    return request.param


@pytest.fixture
def server_class(server_type):
    """Get the server class for the current server type."""
    if server_type == "mcp":
        from gym_mcp_server.servers.mcp import GymMCPServer
        return GymMCPServer
    elif server_type == "http":
        from gym_mcp_server.servers.http import GymHTTPServer
        return GymHTTPServer
    elif server_type == "grpc":
        from gym_mcp_server.servers.grpc import GymGRPCServer
        return GymGRPCServer
    else:
        raise ValueError(f"Unknown server type: {server_type}")


@pytest.fixture
def server_instance(server_type, server_class, mock_env, mock_proto_modules):
    """Create a server instance for the current server type."""
    with patch("gymnasium.make", return_value=mock_env):
        if server_type == "grpc":
            mock_pb2, mock_pb2_grpc = mock_proto_modules
            with patch("gym_mcp_server.servers.grpc.gym_service_pb2", mock_pb2), \
                 patch("gym_mcp_server.servers.grpc.gym_service_pb2_grpc", mock_pb2_grpc):
                server = server_class(env_id="CartPole-v1")
                return server, mock_pb2, mock_pb2_grpc
        else:
            server = server_class(env_id="CartPole-v1")
            return server, None, None


@pytest.fixture
def server(server_instance):
    """Extract just the server instance (for convenience)."""
    server, _, _ = server_instance
    return server
