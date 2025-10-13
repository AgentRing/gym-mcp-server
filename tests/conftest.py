"""
Pytest configuration and fixtures for gym-mcp-server tests.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch


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
