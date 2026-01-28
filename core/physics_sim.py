# Author: Bradley R. Kinnard
# physics simulation - causal counterfactual reasoning

from typing import Any

import numpy as np
from scipy import integrate

from utils.helpers import get_logger


logger = get_logger(__name__)


def simulate_causal_counterfactual(
    belief: dict[str, Any],
    world_state: dict[str, Any],
    seed: int = 42
) -> dict[str, Any]:
    """
    simulate causal counterfactual for a belief given world state.
    uses scipy for mathematical modeling of causal dynamics.
    deterministic with fixed seed.
    """
    np.random.seed(seed)

    # extract belief confidence as initial condition
    y0 = belief.get("confidence", 0.5)

    # simple causal dynamics: dy/dt = -decay * y + influence
    decay = world_state.get("decay_rate", 0.1)
    influence = world_state.get("influence", 0.05)

    def dynamics(t, y):
        return -decay * y + influence

    # integrate over time span
    t_span = (0, 1)
    solution = integrate.solve_ivp(dynamics, t_span, [y0], dense_output=True)

    # final value
    final_confidence = float(solution.y[0, -1])
    final_confidence = max(0.0, min(1.0, final_confidence))

    delta = final_confidence - y0

    logger.info(f"causal sim: {y0:.3f} -> {final_confidence:.3f}, delta={delta:.3f}")

    return {
        "belief_id": belief.get("id", "unknown"),
        "initial_confidence": y0,
        "final_confidence": final_confidence,
        "delta": delta,
        "seed": seed,
        "dynamics": {"decay": decay, "influence": influence}
    }


def simulate_physics_environment(
    initial_state: dict[str, float],
    actions: list[dict[str, float]],
    seed: int = 42
) -> list[dict[str, float]]:
    """
    simulate physics environment for offline RL.
    returns trajectory of states.
    """
    np.random.seed(seed)

    trajectory = [initial_state.copy()]
    current = initial_state.copy()

    for action in actions:
        # simple physics: position += velocity * dt + action
        dt = action.get("dt", 0.1)
        force = action.get("force", 0.0)

        current["velocity"] = current.get("velocity", 0.0) + force * dt
        current["position"] = current.get("position", 0.0) + current["velocity"] * dt

        trajectory.append(current.copy())

    logger.debug(f"simulated {len(actions)} physics steps")
    return trajectory


def compute_causal_accuracy(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]]
) -> float:
    """compute accuracy of causal predictions vs ground truth."""
    if len(predictions) != len(ground_truth):
        raise ValueError("prediction and ground truth length mismatch")

    if not predictions:
        return 1.0

    correct = 0
    for pred, truth in zip(predictions, ground_truth):
        # compare direction of change
        pred_direction = "increase" if pred.get("delta", 0) > 0 else "decrease"
        truth_direction = "increase" if truth.get("delta", 0) > 0 else "decrease"
        if pred_direction == truth_direction:
            correct += 1

    accuracy = correct / len(predictions)
    logger.info(f"causal accuracy: {accuracy:.2%} ({correct}/{len(predictions)})")
    return accuracy
