"""Tests for the BasicRunManager class."""

import pytest
from gym_mcp_server.service import GymService
from gym_mcp_server.run_manager import (
    BasicRunManager,
    RunState,
    EpisodeStatistics,
    RunStatistics,
)

pytestmark = pytest.mark.timeout(10)


class TestBasicRunManager:
    """Test cases for BasicRunManager."""

    def test_init(self):
        """Test run manager initialization."""
        service = GymService(env_id="CartPole-v1", enable_run_manager=False)
        manager = BasicRunManager(
            gym_service=service,
            num_episodes=5,
            max_steps_per_episode=100,
        )
        assert manager.num_episodes == 5
        assert manager.max_steps_per_episode == 100
        assert manager.stats.state == RunState.IDLE
        assert manager.current_episode_stats is None
        service.close()

    def test_start_run(self):
        """Test starting a run."""
        service = GymService(env_id="CartPole-v1", enable_run_manager=False)
        manager = BasicRunManager(
            gym_service=service,
            num_episodes=3,
            max_steps_per_episode=50,
        )
        result = manager.start_run()
        assert result["state"] == RunState.RUNNING.value
        assert manager.stats.state == RunState.RUNNING
        assert manager.stats.num_episodes == 3
        service.close()

    def test_start_episode(self):
        """Test starting an episode."""
        service = GymService(env_id="CartPole-v1", enable_run_manager=False)
        manager = BasicRunManager(
            gym_service=service,
            num_episodes=3,
            max_steps_per_episode=50,
        )
        manager.start_run()
        episode_info = manager.start_episode(seed=42)
        assert episode_info["episode_num"] == 1
        assert episode_info["total_episodes"] == 3
        assert manager.current_episode_stats is not None
        assert manager.current_episode_stats.episode_num == 1
        service.close()

    def test_record_step(self):
        """Test recording a step."""
        service = GymService(env_id="CartPole-v1", enable_run_manager=False)
        manager = BasicRunManager(
            gym_service=service,
            num_episodes=2,
            max_steps_per_episode=10,
        )
        manager.start_run()
        manager.start_episode()

        step_info = manager.record_step(
            action=0,
            observation=[0.0, 0.0, 0.0, 0.0],
            reward=1.0,
            done=False,
            truncated=False,
            info={},
        )

        assert step_info["step_num"] == 1
        assert step_info["episode_num"] == 1
        assert step_info["total_steps"] == 1
        assert step_info["total_reward"] == 1.0
        assert manager.current_episode_stats.total_steps == 1
        assert manager.current_episode_stats.total_reward == 1.0
        service.close()


class TestGymServiceWithRunManager:
    """Test GymService integration with RunManager."""

    def test_service_with_run_manager_enabled(self):
        """Test GymService with run manager enabled."""
        service = GymService(
            env_id="CartPole-v1",
            enable_run_manager=True,
            num_episodes=3,
            max_steps_per_episode=10,
        )
        assert service.run_manager is not None
        assert service.run_manager.stats.state == RunState.RUNNING
        service.close()

    def test_reset_integrates_with_run_manager(self):
        """Test that reset() integrates with run manager."""
        service = GymService(
            env_id="CartPole-v1",
            enable_run_manager=True,
            num_episodes=2,
            max_steps_per_episode=5,
        )

        result = service.reset()
        assert "run_progress" in result
        assert result["run_progress"]["episode_num"] == 1
        assert service.run_manager.current_episode_stats is not None
        service.close()
