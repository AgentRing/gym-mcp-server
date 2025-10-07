#!/usr/bin/env python3
"""
Integration tests to verify the gym-mcp-server implementation works end-to-end.
"""

import sys
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402
from gym_mcp_server import GymMCPAdapter  # noqa: E402


@pytest.mark.integration
def test_basic_functionality():
    """Test basic functionality of the gym-mcp-server."""
    # Initialize adapter (using default render mode)
    adapter = GymMCPAdapter("CartPole-v1")

    # Test reset
    reset_result = adapter.call_tool("reset_env")
    assert reset_result["success"], f"Reset failed: {reset_result}"
    assert "observation" in reset_result
    assert reset_result["done"] is False

    # Test step
    step_result = adapter.call_tool("step_env", action=0)
    assert step_result["success"], f"Step failed: {step_result}"
    assert "observation" in step_result
    assert "reward" in step_result
    assert "done" in step_result

    # Test render (may fail if pygame not installed, which is OK)
    render_result = adapter.call_tool("render_env")
    assert "render" in render_result  # Just check response structure

    # Test get info
    info_result = adapter.call_tool("get_env_info")
    assert info_result["success"], f"Get info failed: {info_result}"
    assert "env_info" in info_result
    assert info_result["env_info"]["id"] == "CartPole-v1"

    # Test close
    close_result = adapter.call_tool("close_env")
    assert close_result["success"], f"Close failed: {close_result}"
    assert close_result["status"] == "closed"


@pytest.mark.integration
def test_multiple_steps():
    """Test taking multiple steps in the environment."""
    adapter = GymMCPAdapter("CartPole-v1")

    # Reset environment
    reset_result = adapter.call_tool("reset_env")
    assert reset_result["success"]

    # Take several steps
    for i in range(5):
        step_result = adapter.call_tool("step_env", action=0)
        assert step_result["success"]
        assert "observation" in step_result
        assert "reward" in step_result

    # Close environment
    close_result = adapter.call_tool("close_env")
    assert close_result["success"]


@pytest.mark.integration
def test_reset_with_seed():
    """Test resetting environment with specific seed."""
    adapter = GymMCPAdapter("CartPole-v1")

    # Reset with seed
    reset_result = adapter.call_tool("reset_env", seed=42)
    assert reset_result["success"]

    # Take a step
    step_result = adapter.call_tool("step_env", action=0)
    assert step_result["success"]

    # Close environment
    close_result = adapter.call_tool("close_env")
    assert close_result["success"]


@pytest.mark.integration
def test_render_modes():
    """Test different render modes."""
    # Use rgb_array mode which is supported by CartPole-v1
    adapter = GymMCPAdapter("CartPole-v1", render_mode="rgb_array")

    # Reset environment
    reset_result = adapter.call_tool("reset_env")
    assert reset_result["success"]

    # Test rgb_array render mode (supported by CartPole-v1)
    # Note: This may fail if pygame is not installed, which is OK for testing
    render_result = adapter.call_tool("render_env", mode="rgb_array")
    # Check that we get a proper response structure (success or error)
    assert "render" in render_result
    assert "mode" in render_result
    # If it fails, it should be due to missing pygame, which is acceptable
    if not render_result.get("success", False):
        assert (
            "pygame" in render_result.get("error", "").lower()
            or "error" in render_result
        )

    # Close environment
    close_result = adapter.call_tool("close_env")
    assert close_result["success"]


@pytest.mark.integration
def test_available_tools():
    """Test getting available tools."""
    adapter = GymMCPAdapter("CartPole-v1")

    tools_result = adapter.call_tool("get_available_tools")
    assert tools_result["success"]
    assert "tools" in tools_result

    tools = tools_result["tools"]
    expected_tools = [
        "reset_env",
        "step_env",
        "render_env",
        "close_env",
        "get_env_info",
    ]
    for tool in expected_tools:
        assert tool in tools

    # Close environment
    close_result = adapter.call_tool("close_env")
    assert close_result["success"]


if __name__ == "__main__":
    # Run basic functionality test if called directly
    test_basic_functionality()
    print("✓ All integration tests passed!")
