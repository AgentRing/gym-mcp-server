"""Tests for utility functions."""

import pytest
import numpy as np
from gym_mcp_server.utils import (
    serialize_observation,
    serialize_action,
    serialize_render_output,
    get_environment_info,
    parse_entry_point,
)
import gymnasium as gym

pytestmark = pytest.mark.timeout(10)


class TestSerialization:
    """Test serialization functions."""

    def test_serialize_observation_int(self):
        """Test serializing integer observation."""
        obs = np.int32(5)
        result = serialize_observation(obs)
        assert result == 5
        assert isinstance(result, int)

    def test_serialize_observation_float(self):
        """Test serializing float observation."""
        obs = np.float32(3.14)
        result = serialize_observation(obs)
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_serialize_observation_bool(self):
        """Test serializing boolean observation."""
        obs = np.bool_(True)
        result = serialize_observation(obs)
        assert result is True
        assert isinstance(result, bool)

    def test_serialize_observation_array(self):
        """Test serializing numpy array observation."""
        obs = np.array([1, 2, 3])
        result = serialize_observation(obs)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_serialize_observation_dict(self):
        """Test serializing dictionary observation."""
        obs = {"key": np.array([1, 2])}
        result = serialize_observation(obs)
        assert result == {"key": [1, 2]}
        assert isinstance(result["key"], list)

    def test_serialize_action(self):
        """Test serializing action."""
        action = np.int32(1)
        result = serialize_action(action)
        assert result == 1
        assert isinstance(result, int)

    def test_serialize_render_output_none(self):
        """Test serializing None render output."""
        result = serialize_render_output(None)
        assert result["render"] is None
        assert "mode" in result

    def test_serialize_render_output_array(self):
        """Test serializing array render output."""
        render_out = np.array([[[255, 0, 0]] * 10] * 10, dtype=np.uint8)
        result = serialize_render_output(render_out, mode="rgb_array")
        assert result["type"] == "image_base64"
        assert "render" in result
        assert result["mode"] == "rgb_array"
        assert "shape" in result

    def test_serialize_render_output_text(self):
        """Test serializing text render output."""
        render_out = "Some text output"
        result = serialize_render_output(render_out, mode="human")
        assert result["type"] == "text"
        assert result["render"] == "Some text output"


class TestEnvironmentInfo:
    """Test environment info functions."""

    def test_get_environment_info(self):
        """Test getting environment information."""
        env = gym.make("CartPole-v1")
        info = get_environment_info(env)
        assert info["id"] == "CartPole-v1"
        assert "action_space" in info
        assert "observation_space" in info
        assert "action_space_size" in info
        assert info["action_space_size"] == 2
        assert "observation_space_shape" in info
        assert info["observation_space_shape"] == [4]
        env.close()


class TestEntryPoint:
    """Test entry point functions."""

    def test_parse_entry_point_valid(self):
        """Test parsing valid entry point."""
        module, class_name = parse_entry_point("mymodule.envs:MyEnv")
        assert module == "mymodule.envs"
        assert class_name == "MyEnv"

    def test_parse_entry_point_invalid(self):
        """Test parsing invalid entry point."""
        with pytest.raises(ValueError):
            parse_entry_point("invalid")

    def test_parse_entry_point_missing_colon(self):
        """Test parsing entry point without colon."""
        with pytest.raises(ValueError):
            parse_entry_point("mymodule.envs")
