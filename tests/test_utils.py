"""
Tests for gym_mcp_server.utils module.
"""

import numpy as np
from unittest.mock import Mock, patch
from gym_mcp_server.utils import (
    serialize_observation,
    serialize_action,
    serialize_render_output,
    get_environment_info,
)


class TestSerializeObservation:
    """Test cases for serialize_observation function."""

    def test_numpy_integer(self):
        """Test serialization of numpy integer."""
        obs = np.int32(42)
        result = serialize_observation(obs)
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_floating(self):
        """Test serialization of numpy floating point."""
        obs = np.float64(3.14)
        result = serialize_observation(obs)
        assert result == 3.14
        assert isinstance(result, float)

    def test_numpy_bool(self):
        """Test serialization of numpy boolean."""
        obs = np.bool_(True)
        result = serialize_observation(obs)
        assert result is True
        assert isinstance(result, bool)

    def test_numpy_array(self):
        """Test serialization of numpy array."""
        obs = np.array([1, 2, 3, 4])
        result = serialize_observation(obs)
        assert result == [1, 2, 3, 4]
        assert isinstance(result, list)

    def test_basic_python_types(self):
        """Test serialization of basic Python types."""
        # Test int
        assert serialize_observation(42) == 42
        # Test float
        assert serialize_observation(3.14) == 3.14
        # Test str
        assert serialize_observation("hello") == "hello"
        # Test bool
        assert serialize_observation(True) is True
        # Test None
        assert serialize_observation(None) is None

    def test_object_with_tolist(self):
        """Test serialization of object with tolist method."""

        class MockArray:
            def tolist(self):
                return [1, 2, 3]

        obs = MockArray()
        result = serialize_observation(obs)
        assert result == [1, 2, 3]

    def test_object_with_item(self):
        """Test serialization of object with item method."""

        class MockScalar:
            def item(self):
                return 42

        obs = MockScalar()
        result = serialize_observation(obs)
        assert result == 42

    def test_dictionary(self):
        """Test serialization of dictionary."""
        obs = {"key1": np.array([1, 2]), "key2": "value"}
        result = serialize_observation(obs)
        expected = {"key1": [1, 2], "key2": "value"}
        assert result == expected

    def test_list(self):
        """Test serialization of list."""
        obs = [np.array([1, 2]), "string", 42]
        result = serialize_observation(obs)
        expected = [[1, 2], "string", 42]
        assert result == expected

    def test_tuple(self):
        """Test serialization of tuple."""
        obs = (np.array([1, 2]), "string", 42)
        result = serialize_observation(obs)
        expected = [[1, 2], "string", 42]
        assert result == expected

    def test_fallback_to_string(self):
        """Test fallback to string representation."""

        class CustomObject:
            def __str__(self):
                return "custom_object"

        obs = CustomObject()
        result = serialize_observation(obs)
        assert result == "custom_object"


class TestSerializeAction:
    """Test cases for serialize_action function."""

    def test_basic_python_types(self):
        """Test serialization of basic Python types."""
        # Test int
        assert serialize_action(42) == 42
        # Test float
        assert serialize_action(3.14) == 3.14
        # Test str
        assert serialize_action("hello") == "hello"
        # Test bool
        assert serialize_action(True) is True
        # Test None
        assert serialize_action(None) is None

    def test_object_with_tolist(self):
        """Test serialization of object with tolist method."""

        class MockArray:
            def tolist(self):
                return [1, 2, 3]

        action = MockArray()
        result = serialize_action(action)
        assert result == [1, 2, 3]

    def test_object_with_item(self):
        """Test serialization of object with item method."""

        class MockScalar:
            def item(self):
                return 42

        action = MockScalar()
        result = serialize_action(action)
        assert result == 42

    def test_dictionary(self):
        """Test serialization of dictionary."""
        action = {"key1": np.array([1, 2]), "key2": "value"}
        result = serialize_action(action)
        expected = {"key1": [1, 2], "key2": "value"}
        assert result == expected

    def test_list(self):
        """Test serialization of list."""
        action = [np.array([1, 2]), "string", 42]
        result = serialize_action(action)
        expected = [[1, 2], "string", 42]
        assert result == expected

    def test_tuple(self):
        """Test serialization of tuple."""
        action = (np.array([1, 2]), "string", 42)
        result = serialize_action(action)
        expected = [[1, 2], "string", 42]
        assert result == expected

    def test_numpy_array(self):
        """Test serialization of numpy array."""
        action = np.array([1, 2, 3, 4])
        result = serialize_action(action)
        assert result == [1, 2, 3, 4]
        assert isinstance(result, list)

    def test_return_unchanged(self):
        """Test that non-special objects are returned unchanged."""

        class CustomObject:
            pass

        action = CustomObject()
        result = serialize_action(action)
        assert result is action


class TestSerializeRenderOutput:
    """Test cases for serialize_render_output function."""

    def test_none_output(self):
        """Test serialization of None render output."""
        result = serialize_render_output(None, "ansi")
        expected = {"render": None, "mode": "ansi"}
        assert result == expected

    def test_rgb_array_uint8(self):
        """Test serialization of RGB array with uint8 dtype."""
        # Create a mock RGB array
        render_out = np.array(
            [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8
        )

        with patch("gym_mcp_server.utils.Image") as mock_image:
            with patch("gym_mcp_server.utils.io") as mock_io:
                # Mock the image processing
                mock_img = Mock()
                mock_image.fromarray.return_value = mock_img
                mock_buffer = Mock()
                mock_buffer.getvalue.return_value = b"mock_image_data"
                mock_io.BytesIO.return_value = mock_buffer
                mock_img.save = Mock()

                result = serialize_render_output(render_out, "rgb_array")

                assert result["mode"] == "rgb_array"
                assert result["type"] == "image_base64"
                assert "render" in result
                assert "shape" in result
                assert result["shape"] == render_out.shape

    def test_rgb_array_non_uint8(self):
        """Test serialization of RGB array with non-uint8 dtype."""
        render_out = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float32)
        result = serialize_render_output(render_out, "rgb_array")

        assert result["mode"] == "rgb_array"
        assert result["type"] == "array"
        assert result["render"] == render_out.tolist()
        assert result["shape"] == render_out.shape

    def test_text_output(self):
        """Test serialization of text render output."""
        render_out = "Rendered environment state"
        result = serialize_render_output(render_out, "ansi")

        assert result["render"] == "Rendered environment state"
        assert result["mode"] == "ansi"
        assert result["type"] == "text"

    def test_object_with_tolist(self):
        """Test serialization of object with tolist method."""

        class MockArray:
            def tolist(self):
                return [1, 2, 3]

        render_out = MockArray()
        result = serialize_render_output(render_out, "ansi")

        assert result["render"] == [1, 2, 3]
        assert result["mode"] == "ansi"
        assert result["type"] == "array"

    def test_fallback_to_string(self):
        """Test fallback to string representation."""

        class CustomObject:
            def __str__(self):
                return "custom_render"

        render_out = CustomObject()
        result = serialize_render_output(render_out, "ansi")

        assert result["render"] == "custom_render"
        assert result["mode"] == "ansi"
        assert result["type"] == "string"


class TestGetEnvironmentInfo:
    """Test cases for get_environment_info function."""

    def test_basic_environment_info(self):
        """Test getting basic environment information."""
        # Create mock environment
        env = Mock()
        env.spec = Mock()
        env.spec.id = "CartPole-v1"
        env.action_space = Mock()
        env.action_space.__str__ = Mock(return_value="Discrete(2)")
        env.action_space.n = 2
        env.observation_space = Mock()
        env.observation_space.__str__ = Mock(return_value="Box(4,)")
        env.observation_space.shape = (4,)
        env.reward_range = (-1.0, 1.0)

        result = get_environment_info(env)

        assert result["id"] == "CartPole-v1"
        assert result["action_space"] == "Discrete(2)"
        assert result["observation_space"] == "Box(4,)"
        assert result["reward_range"] == (-1.0, 1.0)
        assert result["action_space_size"] == 2
        assert result["observation_space_shape"] == [4]

    def test_discrete_action_space(self):
        """Test environment with discrete action space."""
        env = Mock()
        env.spec = Mock()
        env.spec.id = "CartPole-v1"
        env.action_space = Mock()
        env.action_space.__str__ = Mock(return_value="Discrete(2)")
        env.action_space.n = 2
        env.observation_space = Mock()
        env.observation_space.__str__ = Mock(return_value="Box(4,)")
        env.observation_space.shape = (4,)
        env.reward_range = (-1.0, 1.0)

        result = get_environment_info(env)

        assert result["action_space_size"] == 2
        assert result["observation_space_shape"] == [4]

    def test_continuous_action_space(self):
        """Test environment with continuous action space."""
        env = Mock()
        env.spec = Mock()
        env.spec.id = "ContinuousCartPole-v1"
        env.action_space = Mock(spec=["shape", "__str__"])
        env.action_space.__str__ = Mock(return_value="Box(1,)")
        env.action_space.shape = (1,)
        env.observation_space = Mock()
        env.observation_space.__str__ = Mock(return_value="Box(4,)")
        env.observation_space.shape = (4,)
        env.reward_range = (-1.0, 1.0)

        result = get_environment_info(env)

        assert result["action_space_shape"] == [1]
        assert result["observation_space_shape"] == [4]

    def test_observation_space_with_shape(self):
        """Test environment with observation space shape."""
        env = Mock()
        env.spec = Mock()
        env.spec.id = "CartPole-v1"
        env.action_space = Mock()
        env.action_space.__str__ = Mock(return_value="Discrete(2)")
        env.action_space.n = 2
        env.observation_space = Mock()
        env.observation_space.__str__ = Mock(return_value="Box(4,)")
        env.observation_space.shape = (4,)
        env.reward_range = (-1.0, 1.0)

        result = get_environment_info(env)

        assert result["action_space_size"] == 2
        assert result["observation_space_shape"] == [4]

    def test_observation_space_with_spaces(self):
        """Test environment with dictionary observation space."""
        env = Mock()
        env.spec = Mock()
        env.spec.id = "DictCartPole-v1"
        env.action_space = Mock()
        env.action_space.__str__ = Mock(return_value="Discrete(2)")
        env.action_space.n = 2
        env.observation_space = Mock(spec=["spaces", "__str__"])
        env.observation_space.__str__ = Mock(return_value="Dict(obs:Box(4,))")
        mock_obs_space = Mock()
        mock_obs_space.__str__ = Mock(return_value="Box(4,)")
        env.observation_space.spaces = {"obs": mock_obs_space}
        env.reward_range = (-1.0, 1.0)

        result = get_environment_info(env)

        assert result["action_space_size"] == 2
        assert result["observation_space_spaces"] == {"obs": "Box(4,)"}

    def test_environment_without_spec(self):
        """Test environment without spec attribute."""
        env = Mock()
        env.spec = None
        env.action_space = Mock()
        env.action_space.__str__ = Mock(return_value="Discrete(2)")
        env.action_space.n = 2
        env.observation_space = Mock()
        env.observation_space.__str__ = Mock(return_value="Box(4,)")
        env.observation_space.shape = (4,)
        env.reward_range = (-1.0, 1.0)

        result = get_environment_info(env)

        assert result["id"] is None
        assert result["action_space_size"] == 2
        assert result["observation_space_shape"] == [4]

    def test_environment_without_reward_range(self):
        """Test environment without reward_range attribute."""
        env = Mock(spec=["spec", "action_space", "observation_space"])
        env.spec = Mock()
        env.spec.id = "CartPole-v1"
        env.action_space = Mock()
        env.action_space.__str__ = Mock(return_value="Discrete(2)")
        env.action_space.n = 2
        env.observation_space = Mock()
        env.observation_space.__str__ = Mock(return_value="Box(4,)")
        env.observation_space.shape = (4,)
        # No reward_range attribute

        result = get_environment_info(env)

        assert result["action_space_size"] == 2
        assert result["observation_space_shape"] == [4]
        assert result["reward_range"] is None
