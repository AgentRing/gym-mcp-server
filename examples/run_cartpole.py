#!/usr/bin/env python3
"""
Example script showing how to use the Gym MCP adapter with CartPole environment.
"""

import sys
import json
from pathlib import Path

# Add the parent directory to the path so we can import gym_mcp_server
sys.path.insert(0, str(Path(__file__).parent.parent))

from gym_mcp_server import GymMCPAdapter  # noqa: E402


def main():
    """Run a simple CartPole example."""
    print("=== Gym MCP Adapter - CartPole Example ===")

    # Initialize the adapter
    adapter = GymMCPAdapter("CartPole-v1")

    try:
        # Reset the environment
        print("\n1. Resetting environment...")
        reset_result = adapter.call_tool("reset_env")
        print(f"Reset result: {json.dumps(reset_result, indent=2)}")

        if not reset_result.get("success", False):
            print("Failed to reset environment")
            return 1

        obs = reset_result["observation"]
        done = reset_result["done"]
        step_count = 0

        print(f"\n2. Starting episode with initial observation: {obs}")

        # Run a simple episode
        while not done and step_count < 100:  # Limit steps for demo
            # Simple policy: move right if pole angle is positive, left otherwise
            if obs[2] > 0:  # pole angle
                action = 1  # move right
            else:
                action = 0  # move left

            print(f"Step {step_count + 1}: Taking action {action}")

            # Take a step
            step_result = adapter.call_tool("step_env", action=action)
            print(f"Step result: {json.dumps(step_result, indent=2)}")

            if not step_result.get("success", False):
                print("Failed to take step")
                break

            obs = step_result["observation"]
            reward = step_result["reward"]
            done = step_result["done"]
            step_count += 1

            print(f"  Observation: {obs}")
            print(f"  Reward: {reward}")
            print(f"  Done: {done}")

        print(f"\n3. Episode finished after {step_count} steps")

        # Get environment info
        print("\n4. Getting environment information...")
        info_result = adapter.call_tool("get_env_info")
        print(f"Environment info: {json.dumps(info_result, indent=2)}")

        # Render the final state
        print("\n5. Rendering final state...")
        render_result = adapter.call_tool("render_env")
        print(f"Render result: {json.dumps(render_result, indent=2)}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        # Close the environment
        print("\n6. Closing environment...")
        close_result = adapter.call_tool("close_env")
        print(f"Close result: {json.dumps(close_result, indent=2)}")

    return 0


if __name__ == "__main__":
    exit(main())
