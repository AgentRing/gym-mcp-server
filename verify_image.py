#!/usr/bin/env python3
"""
Verification script for Docker images containing gym-mcp-server.

This script verifies that a Docker image properly runs gym-mcp-server
and responds correctly to MCP API calls.

Usage:
    python verify_image.py <image_name> [options]

Examples:
    # Verify a specific image
    python verify_image.py my-gym-server:latest

    # Verify with custom port mapping
    python verify_image.py my-gym-server:latest --host-port 8765

    # Verify with longer timeout
    python verify_image.py my-gym-server:latest --startup-timeout 60
"""

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

try:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
except ImportError:
    print(
        "Error: mcp package not found. Install with: pip install mcp",
        file=sys.stderr,
    )
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Expected tool names
EXPECTED_TOOLS = {
    "reset_env",
    "step_env",
    "render_env",
    "get_env_info",
    "close_env",
}

# Default configuration
DEFAULT_HOST = "localhost"
DEFAULT_CONTAINER_PORT = 8000
DEFAULT_STARTUP_TIMEOUT = 30
DEFAULT_CONTAINER_TIMEOUT = 60


class ImageVerifier:
    """Verifies Docker images running gym-mcp-server."""

    def __init__(
        self,
        image_name: str,
        host: str = DEFAULT_HOST,
        container_port: int = DEFAULT_CONTAINER_PORT,
        host_port: Optional[int] = None,
        startup_timeout: int = DEFAULT_STARTUP_TIMEOUT,
        container_timeout: int = DEFAULT_CONTAINER_TIMEOUT,
        ready_delay: int = 5,
    ):
        """Initialize the verifier.

        Args:
            image_name: Docker image name to verify
            host: Host where container will be accessible
            container_port: Port the server runs on inside the container
            host_port: Port to map on the host (defaults to container_port)
            startup_timeout: Time to wait for server to start (seconds)
            container_timeout: Maximum time to keep container running (seconds)
            ready_delay: Delay after port is open before considering server ready (seconds)
        """
        self.image_name = image_name
        self.host = host
        self.container_port = container_port
        self.host_port = host_port or container_port
        self.startup_timeout = startup_timeout
        self.container_timeout = container_timeout
        self.ready_delay = ready_delay
        self.container_id: Optional[str] = None
        self.results: Dict[str, bool] = {}

    def _run_command(self, cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command."""
        logger.debug(f"Running command: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=check,
            )
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {' '.join(cmd)}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            raise

    @staticmethod
    def check_docker_available() -> bool:
        """Check if Docker is available.

        Returns:
            True if Docker is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["docker", "version"],
                capture_output=True,
                text=True,
                check=False,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def start_container(self) -> str:
        """Start the Docker container and return its ID.

        Returns:
            Container ID

        Raises:
            RuntimeError: If container fails to start
        """
        # Check if Docker is available
        if not self.check_docker_available():
            raise RuntimeError(
                "Docker is not available. Please ensure Docker is installed and running."
            )

        logger.info(f"Starting container from image: {self.image_name}")

        # Run container in detached mode
        cmd = [
            "docker",
            "run",
            "-d",
            "-p",
            f"{self.host_port}:{self.container_port}",
            "--rm",
            self.image_name,
        ]

        result = self._run_command(cmd)
        container_id = result.stdout.strip()
        self.container_id = container_id
        logger.info(f"Container started: {container_id[:12]}")

        return container_id

    def stop_container(self) -> None:
        """Stop the Docker container."""
        if not self.container_id:
            return

        logger.info(f"Stopping container: {self.container_id[:12]}")
        try:
            self._run_command(["docker", "stop", self.container_id], check=False)
        except Exception as e:
            logger.warning(f"Error stopping container: {e}")
        finally:
            self.container_id = None

    def wait_for_server(self) -> bool:
        """Wait for the MCP server to be ready.

        Returns:
            True if server is ready, False otherwise
        """
        import urllib.request
        import urllib.error

        url = f"http://{self.host}:{self.host_port}/mcp"
        logger.info(f"Waiting for server to be ready at {url}...")

        start_time = time.time()
        while time.time() - start_time < self.startup_timeout:
            try:
                # Make an actual HTTP request - the MCP server will respond with
                # a proper error (406 or JSON error) when ready, vs connection refused
                req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    # Any response means server is ready
                    logger.info("✓ Server is ready")
                    return True
            except urllib.error.HTTPError as e:
                # HTTP errors (4xx, 5xx) mean server is responding - it's ready
                # MCP servers typically return 400 "Missing session ID" for plain GET
                if e.code in (400, 406):
                    logger.info("✓ Server is ready")
                    return True
                logger.debug(f"HTTP error {e.code}: {e.reason}")
            except urllib.error.URLError as e:
                # Connection refused, server not ready yet
                logger.debug(f"Connection not ready: {e.reason}")
            except Exception as e:
                logger.debug(f"Wait error: {e}")

            time.sleep(1)

        logger.error(f"Server failed to start within {self.startup_timeout} seconds")
        return False

    async def test_tools_list(self, session: ClientSession) -> bool:
        """Test that all expected tools are available.

        Args:
            session: MCP client session

        Returns:
            True if all tools are available, False otherwise
        """
        logger.info("Testing tools list...")
        try:
            tools_response = await session.list_tools()
            available_tools = {tool.name for tool in tools_response.tools}
            logger.info(f"Available tools: {sorted(available_tools)}")

            missing_tools = EXPECTED_TOOLS - available_tools
            if missing_tools:
                logger.error(f"Missing expected tools: {missing_tools}")
                return False

            extra_tools = available_tools - EXPECTED_TOOLS
            if extra_tools:
                logger.info(f"Extra tools found (not required): {extra_tools}")

            logger.info("✓ All expected tools are available")
            return True
        except Exception as e:
            logger.error(f"Error listing tools: {e}", exc_info=True)
            return False

    async def test_reset_env(self, session: ClientSession) -> bool:
        """Test reset_env tool.

        Args:
            session: MCP client session

        Returns:
            True if reset succeeds, False otherwise
        """
        logger.info("Testing reset_env...")
        try:
            result = await session.call_tool("reset_env", arguments={})
            data = json.loads(result.content[0].text)

            # Validate response structure
            if not data.get("success", False):
                logger.error(f"reset_env failed: {data.get('error', 'Unknown error')}")
                return False

            required_fields = ["observation", "info", "done"]
            for field in required_fields:
                if field not in data:
                    logger.error(f"reset_env missing field: {field}")
                    return False

            if data["done"] is not False:
                logger.warning(f"reset_env returned done={data['done']}, expected False")

            logger.info("✓ reset_env succeeded")
            return True
        except Exception as e:
            logger.error(f"Error testing reset_env: {e}", exc_info=True)
            return False

    async def test_step_env(
        self, session: ClientSession, action: int | str = 0
    ) -> bool:
        """Test step_env tool.

        Args:
            session: MCP client session
            action: Action to take (int for Discrete, str for Text action spaces)

        Returns:
            True if step succeeds, False otherwise
        """
        logger.info(f"Testing step_env with action={action!r}...")
        try:
            result = await session.call_tool("step_env", arguments={"action": action})
            data = json.loads(result.content[0].text)

            # Validate response structure
            if not data.get("success", False):
                logger.error(f"step_env failed: {data.get('error', 'Unknown error')}")
                return False

            required_fields = ["observation", "reward", "done", "info"]
            for field in required_fields:
                if field not in data:
                    logger.error(f"step_env missing field: {field}")
                    return False

            # Validate types
            if not isinstance(data["reward"], (int, float)):
                logger.error(f"step_env reward is not numeric: {type(data['reward'])}")
                return False

            if not isinstance(data["done"], bool):
                logger.error(f"step_env done is not boolean: {type(data['done'])}")
                return False

            logger.info(
                f"✓ step_env succeeded (reward={data['reward']}, done={data['done']})"
            )
            return True
        except Exception as e:
            logger.error(f"Error testing step_env: {e}", exc_info=True)
            return False

    async def test_get_env_info(self, session: ClientSession) -> bool:
        """Test get_env_info tool.

        Args:
            session: MCP client session

        Returns:
            True if get_env_info succeeds, False otherwise
        """
        logger.info("Testing get_env_info...")
        try:
            result = await session.call_tool("get_env_info", arguments={})
            data = json.loads(result.content[0].text)

            # Validate response structure
            if not data.get("success", False):
                logger.error(
                    f"get_env_info failed: {data.get('error', 'Unknown error')}"
                )
                return False

            if "env_info" not in data:
                logger.error("get_env_info missing 'env_info' field")
                return False

            env_info = data["env_info"]
            if not isinstance(env_info, dict):
                logger.error(f"env_info is not a dict: {type(env_info)}")
                return False

            # Check for at least some expected fields
            if "id" in env_info:
                logger.info(f"Environment ID: {env_info['id']}")

            # Store action space type for later use in step_env test
            if "action_space" in env_info:
                self._action_space_str = env_info["action_space"]
                logger.debug(f"Action space: {self._action_space_str}")

            logger.info("✓ get_env_info succeeded")
            return True
        except Exception as e:
            logger.error(f"Error testing get_env_info: {e}", exc_info=True)
            return False

    async def test_render_env(self, session: ClientSession) -> bool:
        """Test render_env tool.

        Args:
            session: MCP client session

        Returns:
            True if render_env succeeds, False otherwise
        """
        logger.info("Testing render_env...")
        try:
            result = await session.call_tool("render_env", arguments={})
            data = json.loads(result.content[0].text)

            # Validate response structure
            if not data.get("success", False):
                # Render might fail if mode is not supported, which is acceptable
                logger.warning(
                    f"render_env returned success=False: {data.get('error', 'Unknown error')}"
                )
                # Don't fail the test if render is not available
                return True

            if "mode" not in data:
                logger.error("render_env missing 'mode' field")
                return False

            logger.info("✓ render_env succeeded")
            return True
        except Exception as e:
            logger.warning(f"Error testing render_env (may be optional): {e}")
            # Render is optional, so we don't fail the test
            return True

    async def test_close_env(self, session: ClientSession) -> bool:
        """Test close_env tool.

        Args:
            session: MCP client session

        Returns:
            True if close_env succeeds, False otherwise
        """
        logger.info("Testing close_env...")
        try:
            result = await session.call_tool("close_env", arguments={})
            data = json.loads(result.content[0].text)

            # Validate response structure
            if not data.get("success", False):
                logger.error(f"close_env failed: {data.get('error', 'Unknown error')}")
                return False

            if "status" not in data:
                logger.error("close_env missing 'status' field")
                return False

            logger.info("✓ close_env succeeded")
            return True
        except Exception as e:
            logger.error(f"Error testing close_env: {e}", exc_info=True)
            return False

    async def run_verification(self) -> Tuple[bool, Dict[str, bool]]:
        """Run the full verification suite.

        Returns:
            Tuple of (overall_success, results_dict)
        """
        logger.info(f"Starting verification for image: {self.image_name}")
        logger.info(f"Server will be accessible at http://{self.host}:{self.host_port}/mcp")

        try:
            # Start container
            self.start_container()

            # Wait for server to be ready
            if not self.wait_for_server():
                return False, {"server_startup": False}

            # Connect to server and run tests
            url = f"http://{self.host}:{self.host_port}/mcp"
            logger.info(f"Connecting to MCP server at {url}...")

            try:
                async with streamablehttp_client(url) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        # Initialize session
                        await session.initialize()
                        logger.info("✓ Connected to MCP server")

                        # Run all tests
                        self.results["tools_list"] = await self.test_tools_list(session)
                        self.results["get_env_info"] = await self.test_get_env_info(session)
                        self.results["reset_env"] = await self.test_reset_env(session)

                        # Determine action type based on action space
                        # Text action spaces need string actions, Discrete needs int
                        action: int | str = 0
                        if hasattr(self, "_action_space_str"):
                            if "Text" in self._action_space_str:
                                # Use a simple no-op string action for Text spaces
                                action = "noop"
                            elif "Discrete" in self._action_space_str:
                                action = 0

                        self.results["step_env"] = await self.test_step_env(
                            session, action=action
                        )
                        self.results["render_env"] = await self.test_render_env(session)
                        self.results["close_env"] = await self.test_close_env(session)

            except Exception as e:
                logger.error(f"Error connecting to or using MCP server: {e}", exc_info=True)
                self.results["connection"] = False
                return False, self.results

            # Determine overall success
            overall_success = all(self.results.values())

            return overall_success, self.results

        finally:
            # Always stop the container
            self.stop_container()

    def print_summary(self, success: bool, results: Dict[str, bool]) -> None:
        """Print verification summary.

        Args:
            success: Overall success status
            results: Individual test results
        """
        print("\n" + "=" * 60)
        print("VERIFICATION SUMMARY")
        print("=" * 60)
        print(f"Image: {self.image_name}")
        print(f"Overall Status: {'✓ PASSED' if success else '✗ FAILED'}")
        print("\nTest Results:")
        for test_name, test_result in sorted(results.items()):
            status = "✓ PASS" if test_result else "✗ FAIL"
            print(f"  {test_name:20s} {status}")
        print("=" * 60)


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify Docker images running gym-mcp-server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify a specific image
  python verify_image.py my-gym-server:latest

  # Verify with custom port mapping
  python verify_image.py my-gym-server:latest --host-port 8765

  # Verify with longer timeout
  python verify_image.py my-gym-server:latest --startup-timeout 60
        """,
    )
    parser.add_argument(
        "image",
        type=str,
        help="Docker image name to verify",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"Host where container will be accessible (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--container-port",
        type=int,
        default=DEFAULT_CONTAINER_PORT,
        help=f"Port the server runs on inside container (default: {DEFAULT_CONTAINER_PORT})",
    )
    parser.add_argument(
        "--host-port",
        type=int,
        default=None,
        help="Port to map on the host (defaults to container-port)",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=DEFAULT_STARTUP_TIMEOUT,
        help=f"Time to wait for server to start in seconds (default: {DEFAULT_STARTUP_TIMEOUT})",
    )
    parser.add_argument(
        "--ready-delay",
        type=int,
        default=5,
        help="Delay in seconds after port opens before connecting (default: 5)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create verifier and run tests
    verifier = ImageVerifier(
        image_name=args.image,
        host=args.host,
        container_port=args.container_port,
        host_port=args.host_port,
        startup_timeout=args.startup_timeout,
        ready_delay=args.ready_delay,
    )

    try:
        success, results = await verifier.run_verification()
        verifier.print_summary(success, results)
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        verifier.stop_container()
        return 130
    except Exception as e:
        logger.error(f"Verification failed with error: {e}", exc_info=True)
        verifier.stop_container()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

