#!/usr/bin/env python3
"""
gRPC client example for Gymnasium MCP server.

This demonstrates running episodes using the gRPC transport.

Requirements:
    pip install gym-mcp-server grpcio

Usage:
    # Automatically start local server and connect (default):
    python grpc_example.py

    # Connect to an external gRPC server:
    python grpc_example.py --external --host localhost --port 50051

    # Use a different environment:
    python grpc_example.py --env MountainCar-v0
"""

import argparse
import logging
from typing import Optional

import grpc

from gym_mcp_server import GymGRPCServer

# Import generated proto code
try:
    from gym_mcp_server.proto import gym_service_pb2
    from gym_mcp_server.proto import gym_service_pb2_grpc
except ImportError:
    print(
        "Error: gRPC proto code not generated. "
        "Run: python scripts/generate_proto.py"
    )
    raise

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_episode_grpc(
    host: str,
    port: int,
    env_id: str,
    num_episodes: int = 3,
    start_server: bool = True,
):
    """
    Run episodes using gRPC transport.

    Args:
        host: Server host
        port: Server port
        env_id: Environment ID
        num_episodes: Number of episodes to run
        start_server: Whether to start a local server in non-blocking mode
    """
    server: Optional[GymGRPCServer] = None

    try:
        # Start local server if requested
        if start_server:
            logger.info(f"Starting local gRPC server for {env_id}...")
            server = GymGRPCServer(
                env_id=env_id,
                render_mode=None,
            )
            server.start(host=host, port=port)

            # Verify server is running
            if not server.is_alive():
                raise RuntimeError("Failed to start local server")

        # Connect to the server
        channel = grpc.insecure_channel(f"{host}:{port}")
        stub = gym_service_pb2_grpc.GymServiceStub(channel)

        logger.info(f"Connected to gRPC server at {host}:{port}")

        # Health check
        health_response = stub.Health(gym_service_pb2.HealthRequest())
        logger.info(f"✓ Server health: {health_response.status}")
        logger.info(f"  Environment: {health_response.env_id}")

        # Get environment info
        info_response = stub.GetInfo(gym_service_pb2.GetInfoRequest())
        if info_response.success:
            logger.info(
                f"Environment info: {info_response.env_info.action_space}"
            )

        # Run episodes
        for episode in range(num_episodes):
            logger.info(f"\n--- Episode {episode + 1} ---")

            # Reset environment
            reset_request = gym_service_pb2.ResetRequest()
            reset_response = stub.Reset(reset_request)

            if not reset_response.success:
                logger.error(f"Reset failed: {reset_response.error}")
                continue

            logger.info("Environment reset successfully")

            # Run until done
            done = False
            total_reward = 0.0
            steps = 0

            while not done and steps < 100:
                # Take random action (0 or 1 for CartPole)
                action = gym_service_pb2.Action(int_value=steps % 2)

                step_request = gym_service_pb2.StepRequest(action=action)
                step_response = stub.Step(step_request)

                if not step_response.success:
                    logger.error(f"Step failed: {step_response.error}")
                    break

                total_reward += step_response.reward
                done = step_response.done
                steps += 1

            logger.info(
                f"Episode {episode + 1}: {steps} steps, "
                f"total reward: {total_reward:.2f}"
            )

        # Close environment
        close_response = stub.Close(gym_service_pb2.CloseRequest())
        if close_response.success:
            logger.info("Environment closed successfully")

    finally:
        # Stop local server if we started it
        if server is not None:
            logger.info("Stopping local server...")
            server.stop()
        if "channel" in locals():
            channel.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Connect to a Gymnasium gRPC server and run episodes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start local server and connect (easiest):
  python grpc_example.py

  # Use a different environment:
  python grpc_example.py --env MountainCar-v0

  # Connect to external server:
  python grpc_example.py --external --host localhost --port 50051

  # Run more episodes:
  python grpc_example.py --episodes 5
        """,
    )
    parser.add_argument(
        "--external",
        action="store_true",
        help="Connect to external server instead of starting local one",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Server port (default: 50051)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",
        help="Gymnasium environment ID (default: CartPole-v1)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run (default: 3)",
    )

    args = parser.parse_args()

    try:
        run_episode_grpc(
            host=args.host,
            port=args.port,
            env_id=args.env,
            num_episodes=args.episodes,
            start_server=not args.external,
        )
        logger.info("\n✓ All episodes completed successfully!")
        return 0
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Error running client: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())

