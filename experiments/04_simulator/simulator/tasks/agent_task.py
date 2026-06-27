"""Base class for multi-turn Agent RL tasks.

Agent RL tasks differ from single-turn tasks:
- The agent interacts with an environment over multiple turns (episode)
- Each turn: agent observes state → generates action → environment returns observation + reward
- Training happens on the collected trace (all turns in an episode)
- Reward can be sparse (end-of-episode) or dense (per-step)
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .base import Task, TaskPrompt


@dataclass
class StepResult:
    """Result from one environment step."""

    observation: str  # Text observation from environment
    reward: float  # Immediate reward for this step
    done: bool  # Whether the episode is finished
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeTrace:
    """A complete multi-turn episode trace for training."""

    # Each turn stores the full conversation prefix (prompt) and the agent's response
    turns: List[Dict[str, Any]] = field(default_factory=list)
    # turn dict: {"prompt_tokens", "response_tokens", "logprobs", "step_reward"}
    total_reward: float = 0.0
    num_turns: int = 0
    success: bool = False


class AgentTask(Task):
    """Base class for multi-turn Agent RL tasks.

    Subclasses must implement:
    - load_dataset(): Load the underlying dataset
    - reset_episode(): Start a new episode and return the initial prompt
    - step(action_text): Execute agent action and return StepResult
    - compute_episode_reward(trace): Compute final reward for the episode
    - get_eval_episodes(): Get evaluation episodes
    """

    @abstractmethod
    def reset_episode(self) -> TaskPrompt:
        """Start a new episode from the dataset.

        Returns a TaskPrompt with:
        - text: The initial system + user prompt for the agent (includes tool descriptions)
        - metadata: Task-specific metadata for reward computation
        """

    @abstractmethod
    def step(self, action_text: str, metadata: Dict[str, Any]) -> StepResult:
        """Execute an action in the environment and return the result.

        Args:
            action_text: The agent's raw text output (to be parsed for actions)
            metadata: Episode metadata from reset_episode

        Returns:
            StepResult with observation, reward, done flag, and info dict
        """

    @abstractmethod
    def compute_episode_reward(self, steps: List[StepResult], metadata: Dict[str, Any]) -> float:
        """Compute the final reward for a complete episode.

        This allows sparse reward that depends on the full trajectory,
        e.g., whether the final answer is correct.

        Args:
            steps: List of step results from the episode
            metadata: Episode metadata from reset_episode

        Returns:
            Final episode reward in [0, 1]
        """

    def format_observation(self, step_result: StepResult) -> str:
        """Format an environment observation into text to append to the conversation.

        Override this to customize how observations are presented to the agent.
        Default format wraps observation in <observation> tags.
        """
        return f"\n<observation>\n{step_result.observation}\n</observation>\n"

    # ─── Implement single-turn Task interface for backward compatibility ───

    def get_prompt(self) -> TaskPrompt:
        """For agent tasks, this returns the initial episode prompt.

        Note: The Tenant should use reset_episode() + step() loop instead.
        This exists for API compatibility.
        """
        return self.reset_episode()

    def compute_reward(self, response_text: str, metadata: Dict[str, Any]) -> float:
        """Single-turn reward fallback.

        For agent tasks, episode reward is computed via compute_episode_reward().
        This provides a basic fallback if somehow called in single-turn mode.
        """
        step_result = self.step(response_text, metadata)
        return step_result.reward if step_result.done else 0.0

    @abstractmethod
    def get_eval_prompts(self, n: int) -> List[TaskPrompt]:
        """Get n evaluation prompts (initial prompts for eval episodes)."""
