# Author: Bradley R. Kinnard
# differential privacy mechanisms with formal guarantees

import math
from typing import Any

import numpy as np

from utils.helpers import get_logger

logger = get_logger(__name__)


def laplace_mechanism(
    value: float,
    sensitivity: float,
    epsilon: float,
    seed: int | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    apply laplace mechanism for epsilon-differential privacy.

    theorem: adding Lap(sensitivity/epsilon) noise provides epsilon-DP.

    returns: (noisy_value, metadata)
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if sensitivity < 0:
        raise ValueError("sensitivity must be non-negative")

    if seed is not None:
        np.random.seed(seed)

    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    noisy_value = value + noise

    metadata = {
        "mechanism": "laplace",
        "epsilon": epsilon,
        "sensitivity": sensitivity,
        "scale": scale,
        "noise_magnitude": abs(noise),
        "seed": seed,
    }

    logger.debug(f"laplace: value={value:.4f} -> {noisy_value:.4f}, scale={scale:.4f}")

    return noisy_value, metadata


def gaussian_mechanism(
    value: float,
    sensitivity: float,
    epsilon: float,
    delta: float,
    seed: int | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    apply gaussian mechanism for (epsilon, delta)-differential privacy.

    theorem: adding N(0, sigma^2) noise with sigma = sqrt(2*ln(1.25/delta)) * sensitivity / epsilon
    provides (epsilon, delta)-DP.

    returns: (noisy_value, metadata)
    """
    if epsilon <= 0 or delta <= 0:
        raise ValueError("epsilon and delta must be positive")
    if sensitivity < 0:
        raise ValueError("sensitivity must be non-negative")

    if seed is not None:
        np.random.seed(seed)

    # compute sigma for (epsilon, delta)-DP
    sigma = math.sqrt(2 * math.log(1.25 / delta)) * sensitivity / epsilon
    noise = np.random.normal(0, sigma)
    noisy_value = value + noise

    metadata = {
        "mechanism": "gaussian",
        "epsilon": epsilon,
        "delta": delta,
        "sensitivity": sensitivity,
        "sigma": sigma,
        "noise_magnitude": abs(noise),
        "seed": seed,
    }

    logger.debug(f"gaussian: value={value:.4f} -> {noisy_value:.4f}, sigma={sigma:.4f}")

    return noisy_value, metadata


def clip_l2_norm(
    vector: np.ndarray,
    max_norm: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    clip vector to max L2 norm for sensitivity bounding.

    this is essential for DP-SGD and similar algorithms:
    clipping gradients bounds their contribution.

    returns: (clipped_vector, metadata)
    """
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")

    original_norm = float(np.linalg.norm(vector))

    if original_norm > max_norm:
        clipped = vector * (max_norm / original_norm)
        was_clipped = True
    else:
        clipped = vector.copy()
        was_clipped = False

    metadata = {
        "original_norm": original_norm,
        "max_norm": max_norm,
        "was_clipped": was_clipped,
        "clipped_norm": float(np.linalg.norm(clipped)),
    }

    return clipped, metadata


def randomized_response(
    value: bool,
    epsilon: float,
    seed: int | None = None,
) -> tuple[bool, dict[str, Any]]:
    """
    randomized response for epsilon-DP on boolean values.

    theorem: with probability p = e^epsilon / (1 + e^epsilon), return true value;
    otherwise return random bit. this provides epsilon-DP.

    returns: (response, metadata)
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    if seed is not None:
        np.random.seed(seed)

    # probability of truthful response
    p_truth = math.exp(epsilon) / (1 + math.exp(epsilon))

    if np.random.random() < p_truth:
        response = value
        was_truthful = True
    else:
        response = np.random.random() < 0.5
        was_truthful = False

    metadata = {
        "mechanism": "randomized_response",
        "epsilon": epsilon,
        "p_truth": p_truth,
        "was_truthful": was_truthful,
        "seed": seed,
    }

    return response, metadata


def exponential_mechanism(
    scores: dict[str, float],
    sensitivity: float,
    epsilon: float,
    seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    exponential mechanism for epsilon-DP selection.

    theorem: selecting item i with probability proportional to
    exp(epsilon * score(i) / (2 * sensitivity)) provides epsilon-DP.

    returns: (selected_key, metadata)
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not scores:
        raise ValueError("scores must be non-empty")

    if seed is not None:
        np.random.seed(seed)

    # compute selection probabilities
    keys = list(scores.keys())
    score_values = np.array([scores[k] for k in keys])

    # normalize scores to prevent overflow
    max_score = np.max(score_values)
    adjusted = score_values - max_score

    # compute unnormalized probabilities
    unnorm_probs = np.exp(epsilon * adjusted / (2 * sensitivity))
    probs = unnorm_probs / np.sum(unnorm_probs)

    # select
    idx = np.random.choice(len(keys), p=probs)
    selected = keys[idx]

    metadata = {
        "mechanism": "exponential",
        "epsilon": epsilon,
        "sensitivity": sensitivity,
        "selected": selected,
        "selected_score": scores[selected],
        "probabilities": {k: float(p) for k, p in zip(keys, probs)},
        "seed": seed,
    }

    return selected, metadata


class DPAggregator:
    """
    differentially private aggregation for federated scenarios.

    supports secure aggregation of values with noise calibrated
    to the number of participants.
    """

    def __init__(
        self,
        epsilon: float,
        delta: float = 1e-6,
        clip_bound: float = 1.0,
    ):
        self._epsilon = epsilon
        self._delta = delta
        self._clip_bound = clip_bound

    def aggregate_sum(
        self,
        values: list[float],
        seed: int | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """compute DP sum of values."""
        # clip each value
        clipped = [max(-self._clip_bound, min(self._clip_bound, v)) for v in values]

        # sensitivity is clip_bound (one participant can change sum by at most clip_bound)
        raw_sum = sum(clipped)

        # add noise
        noisy_sum, noise_meta = laplace_mechanism(
            raw_sum,
            sensitivity=self._clip_bound,
            epsilon=self._epsilon,
            seed=seed,
        )

        metadata = {
            "n_values": len(values),
            "raw_sum": raw_sum,
            "noisy_sum": noisy_sum,
            "clip_bound": self._clip_bound,
            **noise_meta,
        }

        return noisy_sum, metadata

    def aggregate_mean(
        self,
        values: list[float],
        seed: int | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """compute DP mean of values."""
        if not values:
            raise ValueError("cannot compute mean of empty list")

        noisy_sum, sum_meta = self.aggregate_sum(values, seed)
        noisy_mean = noisy_sum / len(values)

        metadata = {
            "n_values": len(values),
            "noisy_mean": noisy_mean,
            **sum_meta,
        }

        return noisy_mean, metadata
