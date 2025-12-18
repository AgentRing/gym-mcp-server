"""Tests for the GymService class."""

from gym_mcp_server.service import GymService


class TestGymService:
    """Test cases for GymService."""

    def test_init(self):
        """Test service initialization."""
        service = GymService(env_id="CartPole-v1")
        assert service.env_id == "CartPole-v1"
        assert service.env is not None
        service.close()

    def test_init_with_render_mode(self):
        """Test service initialization with render mode."""
        service = GymService(env_id="CartPole-v1", render_mode="rgb_array")
        assert service.render_mode == "rgb_array"
        service.close()

    def test_reset(self):
        """Test environment reset."""
        service = GymService(env_id="CartPole-v1")
        result = service.reset()
        assert result["success"] is True
        assert "observation" in result
        assert result["done"] is False
        service.close()

    def test_reset_with_seed(self):
        """Test environment reset with seed."""
        service = GymService(env_id="CartPole-v1")
        result1 = service.reset(seed=42)
        result2 = service.reset(seed=42)
        # With same seed, observations should be the same
        assert result1["observation"] == result2["observation"]
        service.close()

    def test_step(self):
        """Test taking a step in the environment."""
        service = GymService(env_id="CartPole-v1")
        service.reset()
        result = service.step(action=0)
        assert result["success"] is True
        assert "observation" in result
        assert "reward" in result
        assert isinstance(result["reward"], float)
        assert "done" in result
        assert isinstance(result["done"], bool)
        service.close()

    def test_step_invalid_action(self):
        """Test step with invalid action."""
        service = GymService(env_id="CartPole-v1")
        service.reset()
        # CartPole has 2 actions (0 or 1), so action 2 should fail
        result = service.step(action=2)
        # The service should handle this gracefully
        assert "success" in result
        service.close()

    def test_render(self):
        """Test rendering the environment."""
        service = GymService(env_id="CartPole-v1", render_mode="rgb_array")
        service.reset()
        result = service.render()
        assert result["success"] is True
        assert "render" in result
        service.close()

    def test_render_with_mode(self):
        """Test rendering with specific mode."""
        service = GymService(env_id="CartPole-v1")
        service.reset()
        result = service.render(mode="rgb_array")
        assert result["success"] is True
        service.close()

    def test_get_info(self):
        """Test getting environment information."""
        service = GymService(env_id="CartPole-v1")
        result = service.get_info()
        assert result["success"] is True
        assert "env_info" in result
        assert result["env_info"]["id"] == "CartPole-v1"
        assert "action_space" in result["env_info"]
        assert "observation_space" in result["env_info"]
        service.close()

    def test_close(self):
        """Test closing the environment."""
        service = GymService(env_id="CartPole-v1")
        result = service.close()
        assert result["success"] is True
        assert result["status"] == "closed"

    def test_multiple_steps(self):
        """Test taking multiple steps."""
        service = GymService(env_id="CartPole-v1")
        service.reset()
        for _ in range(5):
            result = service.step(action=0)
            assert result["success"] is True
        service.close()

    def test_reset_after_step(self):
        """Test resetting after taking steps."""
        service = GymService(env_id="CartPole-v1")
        service.reset()
        service.step(action=0)
        result = service.reset()
        assert result["success"] is True
        service.close()
