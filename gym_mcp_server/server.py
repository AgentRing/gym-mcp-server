"""
Main MCP server implementation for exposing Gymnasium environments as MCP tools.
"""

import gymnasium as gym
import json
import argparse
import logging
from typing import Any, Dict, Optional
from .utils import (
    serialize_observation,
    serialize_action,
    serialize_render_output,
    get_environment_info,
)
from .schemas import TOOL_SCHEMAS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GymMCPServer:
    """
    MCP server that exposes any Gymnasium environment as MCP tools.
    """

    def __init__(self, env_id: str, render_mode: str = "ansi") -> None:
        """
        Initialize the MCP server with a Gymnasium environment.

        Args:
            env_id: The Gymnasium environment ID (e.g., "CartPole-v1")
            render_mode: Default render mode for the environment
        """
        self.env_id = env_id
        self.render_mode = render_mode
        self.env: Any = None
        self._initialize_environment()

    def _initialize_environment(self) -> None:
        """Initialize the Gymnasium environment."""
        try:
            self.env = gym.make(self.env_id, render_mode=self.render_mode)
            logger.info(f"Successfully initialized environment: {self.env_id}")
        except Exception as e:
            logger.error(f"Failed to initialize environment {self.env_id}: {e}")
            raise

    def _serialize_obs(self, obs: Any) -> Any:
        """Convert observations to JSON-safe formats."""
        return serialize_observation(obs)

    def _serialize_action(self, action: Any) -> Any:
        """Convert actions to JSON-safe formats."""
        return serialize_action(action)

    def reset_env(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Reset the environment to its initial state.

        Args:
            seed: Random seed for reproducible episodes (optional)

        Returns:
            Dictionary containing initial observation, info, and done status
        """
        try:
            if seed is not None:
                obs, info = self.env.reset(seed=seed)
            else:
                obs, info = self.env.reset()

            return {
                "observation": self._serialize_obs(obs),
                "info": info,
                "done": False,
                "success": True,
            }
        except Exception as e:
            logger.error(f"Error resetting environment: {e}")
            return {
                "observation": None,
                "info": {},
                "done": True,
                "success": False,
                "error": str(e),
            }

    def step_env(self, action: Any) -> Dict[str, Any]:
        """
        Take an action in the environment.

        Args:
            action: The action to take in the environment

        Returns:
            Dictionary containing next observation, reward, done status, and info
        """
        try:
            # Convert action to appropriate format if needed
            if isinstance(action, (list, tuple)) and len(action) == 1:
                action = action[0]

            obs, reward, done, truncated, info = self.env.step(action)

            return {
                "observation": self._serialize_obs(obs),
                "reward": float(reward),
                "done": bool(done or truncated),
                "truncated": bool(truncated),
                "info": info,
                "success": True,
            }
        except Exception as e:
            logger.error(f"Error taking step: {e}")
            return {
                "observation": None,
                "reward": 0.0,
                "done": True,
                "truncated": False,
                "info": {},
                "success": False,
                "error": str(e),
            }

    def render_env(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Render the current state of the environment.

        Args:
            mode: Render mode (ansi, rgb_array, human, etc.)

        Returns:
            Dictionary containing the rendered output
        """
        try:
            if mode is None:
                mode = self.render_mode

            render_out = self.env.render()
            result = serialize_render_output(render_out, mode)
            result["success"] = True
            return result
        except Exception as e:
            logger.error(f"Error rendering environment: {e}")
            return {
                "render": None,
                "mode": mode or self.render_mode,
                "type": "error",
                "success": False,
                "error": str(e),
            }

    def close_env(self) -> Dict[str, Any]:
        """
        Close the environment and free resources.

        Returns:
            Dictionary containing the close status
        """
        try:
            self.env.close()
            return {"status": "closed", "success": True}
        except Exception as e:
            logger.error(f"Error closing environment: {e}")
            return {"status": "error", "success": False, "error": str(e)}

    def get_env_info(self) -> Dict[str, Any]:
        """
        Get information about the environment.

        Returns:
            Dictionary containing environment metadata
        """
        try:
            return {"env_info": get_environment_info(self.env), "success": True}
        except Exception as e:
            logger.error(f"Error getting environment info: {e}")
            return {"env_info": {}, "success": False, "error": str(e)}

    def get_available_tools(self) -> Dict[str, Any]:
        """
        Get information about available MCP tools.

        Returns:
            Dictionary containing tool schemas
        """
        return {"tools": TOOL_SCHEMAS, "success": True}


class GymMCPAdapter:
    """
    Adapter class that provides a simple interface for MCP clients.
    """

    def __init__(self, env_id: str, render_mode: str = "ansi") -> None:
        """
        Initialize the adapter with a Gymnasium environment.

        Args:
            env_id: The Gymnasium environment ID
            render_mode: Default render mode for the environment
        """
        self.server = GymMCPServer(env_id, render_mode)
        self.env_id = env_id

    def call_tool(self, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Call a specific tool by name.

        Args:
            tool_name: Name of the tool to call
            **kwargs: Arguments for the tool

        Returns:
            Result of the tool call
        """
        if tool_name == "reset_env":
            return self.server.reset_env(**kwargs)
        elif tool_name == "step_env":
            return self.server.step_env(**kwargs)
        elif tool_name == "render_env":
            return self.server.render_env(**kwargs)
        elif tool_name == "close_env":
            return self.server.close_env()
        elif tool_name == "get_env_info":
            return self.server.get_env_info()
        elif tool_name == "get_available_tools":
            return self.server.get_available_tools()
        else:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

    def run_interactive(self) -> None:
        """
        Run an interactive session with the environment.
        """
        print(f"Running interactive Gym MCP adapter for {self.env_id}")
        print("Available commands:")
        print("  reset [seed] - Reset the environment")
        print("  step <action> - Take an action")
        print("  render [mode] - Render the environment")
        print("  info - Get environment information")
        print("  close - Close the environment")
        print("  quit - Exit the program")
        print()

        while True:
            try:
                command = input("> ").strip().split()
                if not command:
                    continue

                cmd = command[0].lower()

                if cmd == "quit":
                    break
                elif cmd == "reset":
                    seed = int(command[1]) if len(command) > 1 else None
                    result = self.call_tool("reset_env", seed=seed)
                    print(f"Reset result: {json.dumps(result, indent=2)}")
                elif cmd == "step":
                    if len(command) < 2:
                        print("Usage: step <action>")
                        continue
                    action = int(command[1])
                    result = self.call_tool("step_env", action=action)
                    print(f"Step result: {json.dumps(result, indent=2)}")
                elif cmd == "render":
                    mode = command[1] if len(command) > 1 else None
                    result = self.call_tool("render_env", mode=mode)
                    print(f"Render result: {json.dumps(result, indent=2)}")
                elif cmd == "info":
                    result = self.call_tool("get_env_info")
                    print(f"Environment info: {json.dumps(result, indent=2)}")
                elif cmd == "close":
                    result = self.call_tool("close_env")
                    print(f"Close result: {json.dumps(result, indent=2)}")
                else:
                    print(f"Unknown command: {cmd}")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


def main() -> int:
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(
        description="Run a Gymnasium environment as an MCP server."
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Gymnasium environment ID (e.g., CartPole-v1)",
    )
    parser.add_argument(
        "--render-mode", type=str, default="ansi", help="Default render mode"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")

    args = parser.parse_args()

    try:
        adapter = GymMCPAdapter(args.env, args.render_mode)

        if args.interactive:
            adapter.run_interactive()
        else:
            print(f"Gym MCP adapter for {args.env} is ready.")
            print(f"Connect to ws://{args.host}:{args.port}")
            print("Use --interactive flag for interactive mode.")

    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
