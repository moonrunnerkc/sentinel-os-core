# Author: Bradley R. Kinnard
# goal collapse - RL-based goal evolution with differential privacy

import time
from typing import Any

import numpy as np

from utils.helpers import get_logger


logger = get_logger(__name__)


class GoalCollapse:
    """
    manages goal hierarchy and evolution via offline RL.
    supports differential privacy in reward signals.
    """

    def __init__(self):
        self._goals: dict[str, dict[str, Any]] = {}
        self._conflicts: set[tuple[str, str]] = set()

    def create_goal(
        self,
        goal_id: str,
        description: str,
        priority: float,
        parent: str | None = None
    ) -> dict[str, Any]:
        """create a goal in the hierarchy."""
        if not (0.0 <= priority <= 1.0):
            raise ValueError("priority must be between 0 and 1")

        if parent and parent not in self._goals:
            raise KeyError(f"parent goal not found: {parent}")

        goal = {
            "id": goal_id,
            "description": description,
            "priority": priority,
            "parent": parent,
            "created_at": time.time(),
            "updated_at": time.time()
        }
        self._goals[goal_id] = goal
        logger.info(f"created goal {goal_id} with priority {priority:.2f}")
        return goal

    def get_goal(self, goal_id: str) -> dict[str, Any]:
        """retrieve goal by id."""
        if goal_id not in self._goals:
            raise KeyError(f"goal not found: {goal_id}")
        return self._goals[goal_id].copy()

    def get_depth(self, goal_id: str) -> int:
        """compute depth in goal hierarchy."""
        if goal_id not in self._goals:
            raise KeyError(f"goal not found: {goal_id}")

        depth = 0
        current = self._goals[goal_id]
        while current["parent"] is not None:
            depth += 1
            current = self._goals[current["parent"]]
        return depth

    def mark_conflicting(self, id_a: str, id_b: str) -> None:
        """mark two goals as conflicting."""
        if id_a not in self._goals or id_b not in self._goals:
            raise KeyError("goal not found")

        pair = tuple(sorted([id_a, id_b]))
        self._conflicts.add(pair)
        logger.info(f"marked conflict: {id_a} <-> {id_b}")

    def find_conflicts(self) -> list[tuple[str, str]]:
        """return all goal conflicts."""
        return list(self._conflicts)

    def update_priority(self, goal_id: str, delta: float) -> None:
        """update goal priority with bounds enforcement."""
        if goal_id not in self._goals:
            raise KeyError(f"goal not found: {goal_id}")

        goal = self._goals[goal_id]
        new_priority = max(0.0, min(1.0, goal["priority"] + delta))
        goal["priority"] = new_priority
        goal["updated_at"] = time.time()
        logger.debug(f"updated {goal_id} priority to {new_priority:.3f}")

    def reset_goal(self, goal_id: str, priority: float) -> None:
        """reset goal to specific priority."""
        if goal_id not in self._goals:
            raise KeyError(f"goal not found: {goal_id}")

        self._goals[goal_id]["priority"] = max(0.0, min(1.0, priority))
        self._goals[goal_id]["updated_at"] = time.time()

    def evolve(self, goal_id: str, iterations: int, seed: int) -> None:
        """evolve goal priority through seeded random walk."""
        if goal_id not in self._goals:
            raise KeyError(f"goal not found: {goal_id}")

        np.random.seed(seed)
        for _ in range(iterations):
            delta = np.random.uniform(-0.05, 0.05)
            self.update_priority(goal_id, delta)

        logger.info(f"evolved {goal_id} for {iterations} iterations with seed {seed}")

    def evolve_until_stable(
        self,
        goal_id: str,
        epsilon: float = 1e-5,
        max_iterations: int = 1000,
        seed: int = 42
    ) -> tuple[bool, int]:
        """evolve until priority changes less than epsilon."""
        if goal_id not in self._goals:
            raise KeyError(f"goal not found: {goal_id}")

        np.random.seed(seed)
        goal = self._goals[goal_id]
        prev_priority = goal["priority"]

        for i in range(max_iterations):
            delta = np.random.uniform(-0.05, 0.05)
            self.update_priority(goal_id, delta)

            current = self._goals[goal_id]["priority"]
            if abs(current - prev_priority) < epsilon:
                logger.info(f"{goal_id} converged at iteration {i+1}")
                return True, i + 1
            prev_priority = current

        logger.warning(f"{goal_id} did not converge after {max_iterations} iterations")
        return False, max_iterations

    def compute_reward(
        self,
        goal_id: str,
        seed: int,
        epsilon: float = 0.1
    ) -> float:
        """compute reward with differential privacy (laplace noise)."""
        if goal_id not in self._goals:
            raise KeyError(f"goal not found: {goal_id}")

        np.random.seed(seed)
        goal = self._goals[goal_id]

        # base reward is priority
        base_reward = goal["priority"]

        # add laplace noise for differential privacy
        # scale = 1/epsilon for epsilon-DP
        scale = 1.0 / epsilon
        noise = np.random.laplace(0, scale)
        noisy_reward = base_reward + noise

        logger.info(f"computed reward for {goal_id}: base={base_reward:.3f}, "
                   f"noisy={noisy_reward:.3f}, epsilon={epsilon}")
        return noisy_reward
