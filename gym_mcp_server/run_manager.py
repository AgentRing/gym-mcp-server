"""
Basic Run Manager for gym-mcp-server.

Manages episode lifecycle, step tracking, and run statistics.
Designed for local development and testing.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .service import GymService

logger = logging.getLogger(__name__)


class RunState(Enum):
    """State of a run."""

    IDLE = "idle"  # No active run
    RUNNING = "running"  # Run in progress
    PAUSED = "paused"  # Run paused (future feature)
    COMPLETED = "completed"  # Run finished
    ERROR = "error"  # Run failed


@dataclass
class StepResult:
    """Result of a single step."""

    step_num: int
    action: Any
    observation: Any
    reward: float
    done: bool
    truncated: bool
    info: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EpisodeStatistics:
    """Statistics for a single episode."""

    episode_num: int
    start_time: datetime
    end_time: Optional[datetime] = None
    total_steps: int = 0
    total_reward: float = 0.0
    final_reward: float = 0.0
    done: bool = False
    truncated: bool = False
    steps: List[StepResult] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Duration of episode in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    @property
    def success(self) -> bool:
        """Whether episode completed successfully."""
        return self.done and not self.truncated


@dataclass
class RunStatistics:
    """Statistics for a complete run."""

    run_id: str
    env_id: str
    num_episodes: int
    max_steps_per_episode: int
    start_time: datetime
    end_time: Optional[datetime] = None

    # Current state
    current_episode: int = 0
    current_step: int = 0
    state: RunState = RunState.IDLE

    # Episode tracking
    episodes: List[EpisodeStatistics] = field(default_factory=list)

    # Aggregated metrics
    total_steps: int = 0
    total_reward: float = 0.0
    successful_episodes: int = 0

    @property
    def completed_episodes(self) -> int:
        """Number of completed episodes."""
        return len(self.episodes)

    @property
    def average_reward(self) -> float:
        """Average reward per episode."""
        if self.completed_episodes == 0:
            return 0.0
        return self.total_reward / self.completed_episodes

    @property
    def average_steps(self) -> float:
        """Average steps per episode."""
        if self.completed_episodes == 0:
            return 0.0
        return self.total_steps / self.completed_episodes

    @property
    def success_rate(self) -> float:
        """Success rate (completed episodes / total episodes)."""
        if self.completed_episodes == 0:
            return 0.0
        return self.successful_episodes / self.completed_episodes

    @property
    def duration(self) -> float:
        """Total run duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "env_id": self.env_id,
            "num_episodes": self.num_episodes,
            "max_steps_per_episode": self.max_steps_per_episode,
            "state": self.state.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "current_episode": self.current_episode,
            "current_step": self.current_step,
            "completed_episodes": self.completed_episodes,
            "total_steps": self.total_steps,
            "total_reward": self.total_reward,
            "average_reward": self.average_reward,
            "average_steps": self.average_steps,
            "success_rate": self.success_rate,
            "successful_episodes": self.successful_episodes,
            "duration": self.duration,
            "episodes": [
                {
                    "episode_num": ep.episode_num,
                    "total_steps": ep.total_steps,
                    "total_reward": ep.total_reward,
                    "final_reward": ep.final_reward,
                    "done": ep.done,
                    "truncated": ep.truncated,
                    "success": ep.success,
                    "duration": ep.duration,
                    "start_time": ep.start_time.isoformat() if ep.start_time else None,
                    "end_time": ep.end_time.isoformat() if ep.end_time else None,
                }
                for ep in self.episodes
            ],
        }


class BasicRunManager:
    """
    Basic Run Manager for gym environments.

    Manages the lifecycle of runs with multiple episodes:
    - Tracks each step and episode
    - Enforces episode limits
    - Collects comprehensive statistics
    - Provides run status and progress

    Designed for local development and testing.
    """

    def __init__(
        self,
        gym_service: "GymService",
        num_episodes: int = 10,
        max_steps_per_episode: int = 1000,
        run_id: Optional[str] = None,
    ):
        """
        Initialize the run manager.

        Args:
            gym_service: GymService instance for environment operations
            num_episodes: Number of episodes in the run (user overrideable)
            max_steps_per_episode: Maximum steps per episode
            run_id: Optional run ID (generated if not provided)
        """
        self.gym_service = gym_service
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode

        # Generate run ID if not provided
        if run_id is None:
            run_id = str(uuid.uuid4())
        self.run_id = run_id

        # Initialize statistics
        self.stats = RunStatistics(
            run_id=run_id,
            env_id=gym_service.env_id,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            start_time=datetime.now(),
            state=RunState.IDLE,
        )

        # Current episode state
        self.current_episode_stats: Optional[EpisodeStatistics] = None

        logger.info(
            f"Initialized RunManager: run_id={run_id}, "
            f"episodes={num_episodes}, max_steps={max_steps_per_episode}"
        )

    def start_run(self) -> Dict[str, Any]:
        """
        Start a new run.

        Can be called from IDLE or COMPLETED state.
        If called from COMPLETED, it will reset and start a new run.

        Returns:
            Dictionary with run information
        """
        # Allow starting from IDLE or COMPLETED state
        if self.stats.state not in (RunState.IDLE, RunState.COMPLETED):
            raise RuntimeError(
                f"Cannot start run: current state is {self.stats.state.value}. "
                f"Must be IDLE or COMPLETED."
            )

        # If starting from COMPLETED, reset first
        if self.stats.state == RunState.COMPLETED:
            self.reset_run()

        self.stats.state = RunState.RUNNING
        self.stats.start_time = datetime.now()

        logger.info(f"Started run: {self.run_id}")

        return {
            "run_id": self.run_id,
            "state": self.stats.state.value,
            "num_episodes": self.num_episodes,
            "max_steps_per_episode": self.max_steps_per_episode,
            "start_time": self.stats.start_time.isoformat(),
        }

    def start_episode(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Start a new episode.

        This should be called when reset_env() is called.
        The server enforces that reset starts a new episode.

        Args:
            seed: Optional random seed for reproducibility

        Returns:
            Dictionary with episode information
        """
        # Auto-start run if in IDLE state (for gymnasium API compatibility)
        # This allows clients to call reset() without explicitly calling
        # start_run() first
        if self.stats.state == RunState.IDLE:
            logger.info(
                "Auto-starting run from IDLE state (gymnasium API compatibility)"
            )
            self.start_run()

        # Auto-reset and start new run if completed (for gymnasium API compatibility)
        # This allows clients to continue running episodes beyond the configured limit
        if (
            self.stats.state == RunState.COMPLETED
            or self.stats.completed_episodes >= self.num_episodes
        ):
            logger.info(
                f"Run completed ({self.stats.completed_episodes}/"
                f"{self.num_episodes} episodes). "
                "Auto-resetting and starting new run "
                "(gymnasium API compatibility)"
            )
            self.reset_run()
            self.start_run()

        if self.stats.state != RunState.RUNNING:
            raise RuntimeError(
                f"Cannot start episode: run state is {self.stats.state.value}"
            )

        # If there's a current episode, finalize it
        if self.current_episode_stats is not None:
            self._finalize_episode()

        # Start new episode
        self.stats.current_episode += 1
        episode_num = self.stats.current_episode

        self.current_episode_stats = EpisodeStatistics(
            episode_num=episode_num,
            start_time=datetime.now(),
        )

        self.stats.current_step = 0

        logger.info(f"Started episode {episode_num}/{self.num_episodes}")

        return {
            "episode_num": episode_num,
            "total_episodes": self.num_episodes,
            "max_steps": self.max_steps_per_episode,
            "seed": seed,
        }

    def record_step(
        self,
        action: Any,
        observation: Any,
        reward: float,
        done: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Record a step in the current episode.

        This should be called when step_env() is called.
        The server tracks each step automatically.

        Args:
            action: Action taken
            observation: Observation received
            reward: Reward received
            done: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional info dictionary

        Returns:
            Dictionary with step information and run status
        """
        if self.current_episode_stats is None:
            raise RuntimeError("Cannot record step: no active episode")

        if self.stats.state != RunState.RUNNING:
            raise RuntimeError(
                f"Cannot record step: run state is {self.stats.state.value}"
            )

        # Store reference to current episode stats to avoid issues if it becomes None
        # during execution (shouldn't happen, but defensive programming)
        episode_stats = self.current_episode_stats
        if episode_stats is None:
            raise RuntimeError("Cannot record step: no active episode")

        # Check step limit
        if episode_stats.total_steps >= self.max_steps_per_episode:
            logger.warning(
                f"Episode {self.stats.current_episode} exceeded max steps "
                f"({self.max_steps_per_episode})"
            )
            truncated = True
            done = True

        # Increment step counters
        self.stats.current_step += 1
        episode_stats.total_steps += 1
        self.stats.total_steps += 1

        # Update rewards
        episode_stats.total_reward += reward
        episode_stats.final_reward = reward
        self.stats.total_reward += reward

        # Create step result
        step_result = StepResult(
            step_num=episode_stats.total_steps,
            action=action,
            observation=observation,
            reward=reward,
            done=done,
            truncated=truncated,
            info=info,
        )

        episode_stats.steps.append(step_result)

        # Update episode state
        episode_stats.done = done
        episode_stats.truncated = truncated

        # Capture episode stats before potentially finalizing episode
        # (episode will be finalized in _finalize_episode, setting
        # current_episode_stats to None)
        episode_total_steps = episode_stats.total_steps
        episode_total_reward = episode_stats.total_reward

        # Check if episode is complete
        if done or truncated:
            self._finalize_episode()

            # Check if run is complete
            if self.stats.completed_episodes >= self.num_episodes:
                self._finalize_run()

        return {
            "step_num": step_result.step_num,
            "episode_num": self.stats.current_episode,
            "total_steps": episode_total_steps,
            "total_reward": episode_total_reward,
            "done": done,
            "truncated": truncated,
            "run_progress": {
                "completed_episodes": self.stats.completed_episodes,
                "total_episodes": self.num_episodes,
                "run_state": self.stats.state.value,
            },
        }

    def _finalize_episode(self) -> None:
        """Finalize the current episode."""
        if self.current_episode_stats is None:
            return

        self.current_episode_stats.end_time = datetime.now()

        # Check if episode was successful
        if self.current_episode_stats.success:
            self.stats.successful_episodes += 1

        # Add to episodes list
        self.stats.episodes.append(self.current_episode_stats)

        logger.info(
            f"Completed episode {self.current_episode_stats.episode_num}: "
            f"steps={self.current_episode_stats.total_steps}, "
            f"reward={self.current_episode_stats.total_reward:.2f}, "
            f"success={self.current_episode_stats.success}"
        )

        self.current_episode_stats = None

    def _finalize_run(self) -> None:
        """Finalize the run."""
        self.stats.end_time = datetime.now()
        self.stats.state = RunState.COMPLETED

        logger.info(
            f"Completed run {self.run_id}: "
            f"episodes={self.stats.completed_episodes}, "
            f"avg_reward={self.stats.average_reward:.2f}, "
            f"success_rate={self.stats.success_rate:.2%}"
        )

    def get_status(self) -> Dict[str, Any]:
        """
        Get current run status.

        Returns:
            Dictionary with current run status
        """
        return {
            "run_id": self.run_id,
            "state": self.stats.state.value,
            "current_episode": self.stats.current_episode,
            "current_step": self.stats.current_step,
            "completed_episodes": self.stats.completed_episodes,
            "total_episodes": self.num_episodes,
            "progress": {
                "episodes": f"{self.stats.completed_episodes}/{self.num_episodes}",
                "percentage": (
                    self.stats.completed_episodes / self.num_episodes * 100
                    if self.num_episodes > 0
                    else 0
                ),
            },
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get complete run statistics.

        Returns:
            Dictionary with all run statistics
        """
        return self.stats.to_dict()

    def reset_run(self) -> None:
        """
        Reset the run manager for a new run.

        This clears all statistics and resets state.
        Generates a new run_id for the new run cycle.
        """
        # Generate a new run_id for the new run cycle
        new_run_id = str(uuid.uuid4())
        self.run_id = new_run_id

        self.stats = RunStatistics(
            run_id=new_run_id,
            env_id=self.gym_service.env_id,
            num_episodes=self.num_episodes,
            max_steps_per_episode=self.max_steps_per_episode,
            start_time=datetime.now(),
            state=RunState.IDLE,
        )
        self.current_episode_stats = None

        logger.info(f"Reset run manager: new run_id={new_run_id}")
