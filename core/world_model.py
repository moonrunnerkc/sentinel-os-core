# Author: Bradley R. Kinnard
# world model - optional causal simulation for what-if analysis

import time
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

import numpy as np

from utils.helpers import get_logger

logger = get_logger(__name__)


class WorldModelType(Enum):
    """types of world models."""
    NONE = "none"
    SIMPLE = "simple"


@dataclass
class SimulationState:
    """state for world model simulation."""
    beliefs: dict[str, float]  # belief_id -> confidence
    goals: dict[str, float]    # goal_id -> priority
    resources: dict[str, float]  # resource_id -> amount
    step: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "beliefs": self.beliefs.copy(),
            "goals": self.goals.copy(),
            "resources": self.resources.copy(),
            "step": self.step,
        }

    def digest(self) -> str:
        """compute deterministic hash of state."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def copy(self) -> "SimulationState":
        """create deep copy."""
        return SimulationState(
            beliefs=self.beliefs.copy(),
            goals=self.goals.copy(),
            resources=self.resources.copy(),
            step=self.step,
        )


@dataclass
class SimulationAction:
    """action to apply in simulation."""
    action_type: str  # update_belief | update_goal | consume_resource
    target_id: str
    delta: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """result of a simulation run."""
    initial_state: SimulationState
    final_state: SimulationState
    actions: list[SimulationAction]
    reward: float
    steps: int
    seed: int
    deterministic: bool
    initial_digest: str
    final_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_state": self.initial_state.to_dict(),
            "final_state": self.final_state.to_dict(),
            "actions_count": len(self.actions),
            "reward": self.reward,
            "steps": self.steps,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "initial_digest": self.initial_digest,
            "final_digest": self.final_digest,
        }


class WorldModelDisabledError(Exception):
    """raised when world model is called but disabled."""
    pass


class WorldModelInterface(ABC):
    """
    abstract interface for world models.

    all implementations must be:
    - deterministic under fixed seed
    - offline-capable
    - auditable (state digests logged)
    """

    @property
    @abstractmethod
    def model_type(self) -> WorldModelType:
        """return model type."""
        pass

    @abstractmethod
    def simulate(
        self,
        initial_state: SimulationState,
        actions: list[SimulationAction],
        seed: int,
    ) -> SimulationResult:
        """
        simulate actions from initial state.

        must be deterministic given seed.
        """
        pass

    @abstractmethod
    def counterfactual(
        self,
        state: SimulationState,
        action: SimulationAction,
        seed: int,
    ) -> tuple[SimulationState, float]:
        """
        simulate single action and return new state + reward.

        for what-if analysis.
        """
        pass


class SimpleWorldModel(WorldModelInterface):
    """
    simple deterministic world model using numpy.

    models:
    - belief confidence decay over time
    - goal priority adjustment
    - resource consumption and regeneration

    no external dependencies beyond numpy.
    """

    def __init__(
        self,
        decay_rate: float = 0.01,
        regeneration_rate: float = 0.005,
        noise_scale: float = 0.01,
    ):
        self._decay_rate = decay_rate
        self._regeneration_rate = regeneration_rate
        self._noise_scale = noise_scale

    @property
    def model_type(self) -> WorldModelType:
        return WorldModelType.SIMPLE

    def _apply_action(
        self,
        state: SimulationState,
        action: SimulationAction,
        rng: np.random.Generator,
    ) -> tuple[SimulationState, float]:
        """apply single action to state, return new state and reward."""
        new_state = state.copy()
        reward = 0.0

        if action.action_type == "update_belief":
            if action.target_id in new_state.beliefs:
                old_val = new_state.beliefs[action.target_id]
                new_val = np.clip(old_val + action.delta, 0.0, 1.0)
                new_state.beliefs[action.target_id] = new_val
                # reward for increasing confidence
                reward = (new_val - old_val) * 0.1

        elif action.action_type == "update_goal":
            if action.target_id in new_state.goals:
                old_val = new_state.goals[action.target_id]
                new_val = np.clip(old_val + action.delta, 0.0, 1.0)
                new_state.goals[action.target_id] = new_val
                reward = abs(action.delta) * 0.05

        elif action.action_type == "consume_resource":
            if action.target_id in new_state.resources:
                old_val = new_state.resources[action.target_id]
                consumed = min(old_val, abs(action.delta))
                new_state.resources[action.target_id] = old_val - consumed
                reward = consumed * 0.2

        new_state.step += 1
        return new_state, reward

    def _apply_dynamics(
        self,
        state: SimulationState,
        rng: np.random.Generator,
    ) -> SimulationState:
        """apply world dynamics (decay, regeneration)."""
        new_state = state.copy()

        # belief decay
        for bid in new_state.beliefs:
            noise = rng.normal(0, self._noise_scale)
            decay = self._decay_rate + noise
            new_state.beliefs[bid] = max(0.0, new_state.beliefs[bid] - decay)

        # resource regeneration
        for rid in new_state.resources:
            noise = rng.normal(0, self._noise_scale)
            regen = self._regeneration_rate + noise
            new_state.resources[rid] = min(1.0, new_state.resources[rid] + regen)

        return new_state

    def simulate(
        self,
        initial_state: SimulationState,
        actions: list[SimulationAction],
        seed: int,
    ) -> SimulationResult:
        """simulate sequence of actions."""
        rng = np.random.default_rng(seed)
        initial_digest = initial_state.digest()

        current = initial_state.copy()
        total_reward = 0.0

        for action in actions:
            current, reward = self._apply_action(current, action, rng)
            total_reward += reward
            current = self._apply_dynamics(current, rng)

        final_digest = current.digest()

        result = SimulationResult(
            initial_state=initial_state,
            final_state=current,
            actions=actions,
            reward=total_reward,
            steps=len(actions),
            seed=seed,
            deterministic=True,
            initial_digest=initial_digest,
            final_digest=final_digest,
        )

        logger.debug(
            f"simulation complete: steps={len(actions)}, reward={total_reward:.4f}, "
            f"digest={initial_digest[:8]}...{final_digest[:8]}"
        )

        return result

    def counterfactual(
        self,
        state: SimulationState,
        action: SimulationAction,
        seed: int,
    ) -> tuple[SimulationState, float]:
        """simulate single action for what-if analysis."""
        rng = np.random.default_rng(seed)
        new_state, reward = self._apply_action(state.copy(), action, rng)
        new_state = self._apply_dynamics(new_state, rng)
        return new_state, reward


class DisabledWorldModel(WorldModelInterface):
    """
    world model that hard-fails on any operation.

    used when world_model.enabled=false.
    """

    @property
    def model_type(self) -> WorldModelType:
        return WorldModelType.NONE

    def simulate(
        self,
        initial_state: SimulationState,
        actions: list[SimulationAction],
        seed: int,
    ) -> SimulationResult:
        raise WorldModelDisabledError(
            "world model is disabled. set world_model.enabled=true to use."
        )

    def counterfactual(
        self,
        state: SimulationState,
        action: SimulationAction,
        seed: int,
    ) -> tuple[SimulationState, float]:
        raise WorldModelDisabledError(
            "world model is disabled. set world_model.enabled=true to use."
        )


def create_world_model(config: dict[str, Any]) -> WorldModelInterface:
    """
    create world model from config.

    returns DisabledWorldModel if disabled (hard-fails on use).
    """
    wm_config = config.get("world_model", {})

    if not wm_config.get("enabled", False):
        logger.info("world model disabled, returning hard-fail stub")
        return DisabledWorldModel()

    model_type = wm_config.get("type", "simple")

    if model_type == "simple":
        logger.info("creating simple world model")
        return SimpleWorldModel(
            decay_rate=wm_config.get("decay_rate", 0.01),
            regeneration_rate=wm_config.get("regeneration_rate", 0.005),
            noise_scale=wm_config.get("noise_scale", 0.01),
        )
    elif model_type == "none":
        return DisabledWorldModel()
    else:
        raise ValueError(f"unknown world model type: {model_type}")


class WorldModelContext:
    """
    context manager for world model operations.

    ensures determinism by managing seeds and logging.
    """

    def __init__(self, model: WorldModelInterface, base_seed: int = 42):
        self._model = model
        self._base_seed = base_seed
        self._simulation_count = 0
        self._history: list[dict[str, Any]] = []

    def simulate_with_audit(
        self,
        initial_state: SimulationState,
        actions: list[SimulationAction],
    ) -> SimulationResult:
        """run simulation with full audit logging."""
        seed = self._base_seed + self._simulation_count
        self._simulation_count += 1

        start = time.perf_counter()
        result = self._model.simulate(initial_state, actions, seed)
        duration_ms = (time.perf_counter() - start) * 1000

        audit_entry = {
            "simulation_id": self._simulation_count,
            "seed": seed,
            "initial_digest": result.initial_digest,
            "final_digest": result.final_digest,
            "steps": result.steps,
            "reward": result.reward,
            "duration_ms": duration_ms,
            "timestamp": time.time(),
        }
        self._history.append(audit_entry)

        logger.info(
            f"simulation {self._simulation_count}: seed={seed}, "
            f"steps={result.steps}, reward={result.reward:.4f}, took={duration_ms:.2f}ms"
        )

        return result

    def counterfactual_branches(
        self,
        state: SimulationState,
        actions: list[SimulationAction],
    ) -> list[tuple[SimulationAction, SimulationState, float]]:
        """
        evaluate multiple counterfactual branches.

        returns list of (action, resulting_state, reward).
        """
        branches = []
        for i, action in enumerate(actions):
            seed = self._base_seed + self._simulation_count + i
            new_state, reward = self._model.counterfactual(state, action, seed)
            branches.append((action, new_state, reward))

        self._simulation_count += len(actions)
        return branches

    def get_audit_history(self) -> list[dict[str, Any]]:
        """return audit history."""
        return self._history.copy()

    def verify_determinism(
        self,
        initial_state: SimulationState,
        actions: list[SimulationAction],
        seed: int,
        expected_digest: str,
    ) -> tuple[bool, str]:
        """verify simulation produces expected result."""
        result = self._model.simulate(initial_state, actions, seed)
        if result.final_digest == expected_digest:
            return True, "determinism verified"
        return False, f"digest mismatch: {result.final_digest} != {expected_digest}"
