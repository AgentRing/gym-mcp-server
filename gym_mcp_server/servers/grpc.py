"""
gRPC server implementation for exposing Gymnasium environments.

This module provides a gRPC server that exposes gym operations via gRPC,
reusing the shared service layer for consistency with MCP and HTTP servers.
"""

import json
import logging
import threading
import time
from typing import Any, Optional

import grpc
from concurrent import futures

from .base import BaseGymServer

# Import generated proto code
# Note: These will be generated from gym_service.proto
try:
    from ..proto import gym_service_pb2
    from ..proto import gym_service_pb2_grpc
except ImportError:
    # Fallback: we'll generate these at runtime or during build
    gym_service_pb2 = None
    gym_service_pb2_grpc = None

logger = logging.getLogger(__name__)


# Define GymServiceServicer conditionally based on proto availability
if gym_service_pb2_grpc is not None:
    class GymServiceServicer(gym_service_pb2_grpc.GymServiceServicer):
        """gRPC servicer implementation for GymService."""

        def __init__(self, service):
            """Initialize the servicer with a gym service instance.

            Args:
                service: The GymService instance to use for operations
            """
            self._service = service

        def _serialize_observation(self, obs: Any) -> gym_service_pb2.Observation:
            """Convert a Python observation to protobuf Observation message."""
            if isinstance(obs, (int, float, bool)):
                scalar = gym_service_pb2.ScalarObservation()
                if isinstance(obs, int):
                    scalar.int_value = obs
                elif isinstance(obs, float):
                    scalar.float_value = obs
                elif isinstance(obs, bool):
                    scalar.bool_value = obs
                return gym_service_pb2.Observation(scalar=scalar)
            elif isinstance(obs, (list, tuple)) or hasattr(obs, "tolist"):
                # Handle arrays
                import numpy as np
                if hasattr(obs, "tolist"):
                    arr = np.array(obs)
                else:
                    arr = np.array(obs)
                array_obs = gym_service_pb2.ArrayObservation(
                    values=arr.flatten().tolist(),
                    shape=list(arr.shape),
                )
                return gym_service_pb2.Observation(array=array_obs)
            elif isinstance(obs, dict):
                # Handle dict observations
                dict_obs = gym_service_pb2.DictObservation()
                for k, v in obs.items():
                    dict_obs.values[k].CopyFrom(self._serialize_observation(v))
                return gym_service_pb2.Observation(dict=dict_obs)
            else:
                # Fallback: convert to JSON string
                scalar = gym_service_pb2.ScalarObservation()
                scalar.string_value = json.dumps(obs)
                return gym_service_pb2.Observation(scalar=scalar)

        def _deserialize_action(self, action_msg: gym_service_pb2.Action) -> Any:
            """Convert a protobuf Action message to Python action."""
            if action_msg.HasField("int_value"):
                return action_msg.int_value
            elif action_msg.HasField("float_value"):
                return action_msg.float_value
            elif action_msg.HasField("array_value"):
                arr = action_msg.array_value
                import numpy as np
                return np.array(arr.values).reshape(arr.shape).tolist()
            elif action_msg.HasField("dict_value"):
                result = {}
                for k, v in action_msg.dict_value.values.items():
                    result[k] = self._deserialize_action(v)
                return result
            else:
                raise ValueError("Invalid action message")

        def Reset(
            self, request: gym_service_pb2.ResetRequest, context: grpc.ServicerContext
        ) -> gym_service_pb2.ResetResponse:
            """Reset the environment to its initial state."""
            seed = request.seed if request.HasField("seed") else None
            result = self._service.reset(seed=seed)

            response = gym_service_pb2.ResetResponse(
                success=result["success"],
                done=result["done"],
            )

            if not result["success"]:
                response.error = result.get("error", "Unknown error")
            else:
                # Serialize observation
                response.observation.CopyFrom(
                    self._serialize_observation(result["observation"])
                )
                # Serialize info as JSON strings
                for k, v in result.get("info", {}).items():
                    try:
                        response.info[k] = json.dumps(v)
                    except (TypeError, ValueError):
                        response.info[k] = str(v)

            return response

        def Step(
            self, request: gym_service_pb2.StepRequest, context: grpc.ServicerContext
        ) -> gym_service_pb2.StepResponse:
            """Take an action in the environment."""
            action = self._deserialize_action(request.action)
            result = self._service.step(action)

            response = gym_service_pb2.StepResponse(
                success=result["success"],
                reward=result["reward"],
                done=result["done"],
                truncated=result["truncated"],
            )

            if not result["success"]:
                response.error = result.get("error", "Unknown error")
            else:
                # Serialize observation
                response.observation.CopyFrom(
                    self._serialize_observation(result["observation"])
                )
                # Serialize info as JSON strings
                for k, v in result.get("info", {}).items():
                    try:
                        response.info[k] = json.dumps(v)
                    except (TypeError, ValueError):
                        response.info[k] = str(v)

            return response

        def Render(
            self, request: gym_service_pb2.RenderRequest, context: grpc.ServicerContext
        ) -> gym_service_pb2.RenderResponse:
            """Render the current state of the environment."""
            mode = request.mode if request.HasField("mode") else None
            result = self._service.render(mode=mode)

            response = gym_service_pb2.RenderResponse(success=result["success"])

            if not result["success"]:
                response.error = result.get("error", "Unknown error")
            else:
                render_val = result.get("render")
                if render_val is not None:
                    if isinstance(render_val, str):
                        response.render = render_val
                    else:
                        response.render = json.dumps(render_val)
                if result.get("mode"):
                    response.mode = result["mode"]
                if result.get("type"):
                    response.type = result["type"]
                if result.get("shape"):
                    response.shape[:] = result["shape"]

            return response

        def GetInfo(
            self, request: gym_service_pb2.GetInfoRequest, context: grpc.ServicerContext
        ) -> gym_service_pb2.GetInfoResponse:
            """Get information about the environment."""
            result = self._service.get_info()

            response = gym_service_pb2.GetInfoResponse(success=result["success"])

            if not result["success"]:
                response.error = result.get("error", "Unknown error")
            else:
                env_info = result["env_info"]
                info = gym_service_pb2.EnvInfo(
                    action_space=env_info.get("action_space", ""),
                    observation_space=env_info.get("observation_space", ""),
                )

                if env_info.get("id"):
                    info.id = env_info["id"]
                if env_info.get("reward_range"):
                    rr = env_info["reward_range"]
                    info.reward_range.min = float(rr[0])
                    info.reward_range.max = float(rr[1])
                if env_info.get("action_space_size"):
                    info.action_space_size = env_info["action_space_size"]
                if env_info.get("action_space_shape"):
                    info.action_space_shape[:] = env_info["action_space_shape"]
                if env_info.get("observation_space_shape"):
                    info.observation_space_shape[:] = env_info["observation_space_shape"]
                if env_info.get("observation_space_spaces"):
                    for k, v in env_info["observation_space_spaces"].items():
                        info.observation_space_spaces[k] = str(v)

                response.env_info.CopyFrom(info)

            return response

        def Close(
            self, request: gym_service_pb2.CloseRequest, context: grpc.ServicerContext
        ) -> gym_service_pb2.CloseResponse:
            """Close the environment and free resources."""
            result = self._service.close()

            response = gym_service_pb2.CloseResponse(
                success=result["success"],
                status=result["status"],
            )
            if result.get("error"):
                response.error = result["error"]
            return response

        def Health(
            self, request: gym_service_pb2.HealthRequest, context: grpc.ServicerContext
        ) -> gym_service_pb2.HealthResponse:
            """Health check endpoint."""
            return gym_service_pb2.HealthResponse(
                status="healthy",
                env_id=self._service.env_id,
                render_mode=self._service.render_mode or "",
            )
else:
    # Fallback: define a dummy class when proto is not available
    class GymServiceServicer:
        """Dummy servicer class when proto code is not generated."""
        
        def __init__(self, service):
            raise ImportError("gRPC proto code not generated")


class GymGRPCServer(BaseGymServer):
    """gRPC Server for Gymnasium environments.

    This class provides a gRPC server that exposes gym operations,
    reusing the shared service layer for consistency.

    Example (blocking):
        >>> server = GymGRPCServer(env_id="CartPole-v1")
        >>> server.run(host="localhost", port=50051)

    Example (non-blocking):
        >>> server = GymGRPCServer(env_id="CartPole-v1")
        >>> server.start(host="localhost", port=50051)  # Start in background
        >>> # ... do other work ...
        >>> server.stop()  # Stop the server
    """

    def __init__(
        self,
        env_id: str,
        render_mode: Optional[str] = None,
    ):
        """Initialize the Gym gRPC Server.

        Args:
            env_id: The Gymnasium environment ID
            render_mode: Optional render mode for the environment
        """
        if gym_service_pb2 is None or gym_service_pb2_grpc is None:
            raise ImportError(
                "gRPC proto code not generated. "
                "Run: python scripts/generate_proto.py"
            )

        super().__init__(env_id=env_id, render_mode=render_mode)

        # Server state
        self._server: Optional[grpc.Server] = None
        self._port: Optional[int] = None

    def run(self, host: str = "[::]", port: int = 50051) -> None:
        """Run the gRPC server (blocking).

        Args:
            host: Host to bind to (default: [::] for IPv6)
            port: Port to bind to (default: 50051)
        """
        logger.info(f"Starting gRPC server on {host}:{port}")

        # Create gRPC server
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        gym_service_pb2_grpc.add_GymServiceServicer_to_server(
            GymServiceServicer(self._service), self._server
        )

        # Bind to port
        listen_addr = f"{host}:{port}"
        self._server.add_insecure_port(listen_addr)
        self._server.start()

        logger.info(f"gRPC server started on {listen_addr}")

        try:
            self._server.wait_for_termination()
        except KeyboardInterrupt:
            logger.info("Shutting down gRPC server...")
            self._server.stop(grace=5)

    def start(self, host: str = "[::]", port: int = 50051) -> None:
        """Start the server in a background thread (non-blocking).

        Args:
            host: Host to bind to (default: [::] for IPv6)
            port: Port to bind to (default: 50051)

        Raises:
            RuntimeError: If server is already running
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Server already started")

        self._port = port

        self._thread = threading.Thread(
            target=self.run, args=(host, port), daemon=True
        )
        self._thread.start()

        # Wait for server to start
        logger.info("Waiting for gRPC server to start...")
        time.sleep(1)

        logger.info(f"gRPC server should be ready at {host}:{port}")

    def stop(self, grace: int = 5) -> None:
        """Stop the server if running in non-blocking mode.

        Args:
            grace: Grace period in seconds for shutdown
        """
        if self._server is not None:
            logger.info("Stopping gRPC server...")
            self._server.stop(grace=grace)
            self._server = None

