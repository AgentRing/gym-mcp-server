#!/usr/bin/env python3
"""
Example showing how to use the Gym MCP adapter as a client.
This simulates how an MCP client would interact with the Gym environment.
"""

import sys
import json
import random
from pathlib import Path

# Add the parent directory to the path so we can import gym_mcp_server
sys.path.insert(0, str(Path(__file__).parent.parent))

from gym_mcp_server import GymMCPAdapter  # noqa: E402


class MCPClient:
    """
    Simulated MCP client that interacts with the Gym MCP adapter.
    """

    def __init__(self, env_id: str):
        """Initialize the MCP client with a Gym environment."""
        self.adapter = GymMCPAdapter(env_id)
        self.env_id = env_id
        self.episode_count = 0
        self.total_reward = 0.0

    def reset_environment(self, seed: int = None) -> dict:
        """Reset the environment and return initial state."""
        result = self.adapter.call_tool("reset_env", seed=seed)
        if result.get("success", False):
            self.episode_count += 1
            self.total_reward = 0.0
        return result

    def take_action(self, action) -> dict:
        """Take an action in the environment."""
        result = self.adapter.call_tool("step_env", action=action)
        if result.get("success", False):
            self.total_reward += result.get("reward", 0.0)
        return result

    def render_environment(self, mode: str = None) -> dict:
        """Render the current state of the environment."""
        return self.adapter.call_tool("render_env", mode=mode)

    def get_environment_info(self) -> dict:
        """Get information about the environment."""
        return self.adapter.call_tool("get_env_info")

    def close_environment(self) -> dict:
        """Close the environment."""
        return self.adapter.call_tool("close_env")

    def run_random_episode(self, max_steps: int = 200) -> dict:
        """Run a single episode with random actions."""
        print(f"Starting episode {self.episode_count + 1} with random actions...")

        # Reset environment
        reset_result = self.reset_environment()
        if not reset_result.get("success", False):
            return {"success": False, "error": "Failed to reset environment"}

        obs = reset_result["observation"]
        done = reset_result["done"]
        step_count = 0

        print(f"Initial observation: {obs}")

        # Run episode
        while not done and step_count < max_steps:
            # Random action
            action = random.randint(0, 1)

            # Take step
            step_result = self.take_action(action)
            if not step_result.get("success", False):
                break

            obs = step_result["observation"]
            reward = step_result["reward"]
            done = step_result["done"]
            step_count += 1

            print(
                f"Step {step_count}: Action={action}, Reward={reward:.2f}, Done={done}"
            )

        episode_info = {
            "episode": self.episode_count,
            "steps": step_count,
            "total_reward": self.total_reward,
            "success": True,
        }

        print(f"Episode finished: {episode_info}")
        return episode_info

    def run_simple_policy_episode(self, max_steps: int = 200) -> dict:
        """Run a single episode with a simple policy."""
        print(f"Starting episode {self.episode_count + 1} with simple policy...")

        # Reset environment
        reset_result = self.reset_environment()
        if not reset_result.get("success", False):
            return {"success": False, "error": "Failed to reset environment"}

        obs = reset_result["observation"]
        done = reset_result["done"]
        step_count = 0

        print(f"Initial observation: {obs}")

        # Run episode with simple policy
        while not done and step_count < max_steps:
            # Simple policy: move right if pole angle is positive, left otherwise
            if obs[2] > 0:  # pole angle
                action = 1  # move right
            else:
                action = 0  # move left

            # Take step
            step_result = self.take_action(action)
            if not step_result.get("success", False):
                break

            obs = step_result["observation"]
            reward = step_result["reward"]
            done = step_result["done"]
            step_count += 1

            print(
                f"Step {step_count}: Action={action}, Reward={reward:.2f}, Done={done}"
            )

        episode_info = {
            "episode": self.episode_count,
            "steps": step_count,
            "total_reward": self.total_reward,
            "success": True,
        }

        print(f"Episode finished: {episode_info}")
        return episode_info


def main():
    """Run MCP client examples."""
    print("=== MCP Client Example ===")

    # Initialize MCP client
    client = MCPClient("CartPole-v1")

    try:
        # Get environment information
        print("\n1. Getting environment information...")
        info_result = client.get_environment_info()
        print(f"Environment info: {json.dumps(info_result, indent=2)}")

        # Run random episodes
        print("\n2. Running random episodes...")
        for i in range(3):
            episode_result = client.run_random_episode(max_steps=50)
            if not episode_result.get("success", False):
                print(f"Episode {i+1} failed: {episode_result}")
                break

        # Run simple policy episodes
        print("\n3. Running simple policy episodes...")
        for i in range(3):
            episode_result = client.run_simple_policy_episode(max_steps=50)
            if not episode_result.get("success", False):
                print(f"Episode {i+1} failed: {episode_result}")
                break

        # Render final state
        print("\n4. Rendering final state...")
        render_result = client.render_environment()
        print(f"Render result: {json.dumps(render_result, indent=2)}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        # Close environment
        print("\n5. Closing environment...")
        close_result = client.close_environment()
        print(f"Close result: {json.dumps(close_result, indent=2)}")

    return 0


if __name__ == "__main__":
    exit(main())
