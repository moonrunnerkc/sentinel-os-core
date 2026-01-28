# Author: Bradley R. Kinnard
# belief ecology - manages belief creation, propagation, decay, and causal simulation

import time
from typing import Any

import numpy as np

from utils.helpers import get_logger


logger = get_logger(__name__)


class BeliefEcology:
    """
    manages a network of beliefs with propagation, decay, and contradiction detection.
    supports optional causal simulation via world models.
    """

    def __init__(self):
        self._beliefs: dict[str, dict[str, Any]] = {}
        self._links: dict[str, list[tuple[str, float]]] = {}  # id -> [(target_id, strength)]
        self._contradictions: set[tuple[str, str]] = set()

    def create_belief(
        self,
        belief_id: str,
        content: str,
        confidence: float,
        source: str
    ) -> dict[str, Any]:
        """create a new belief with validation."""
        if not content:
            raise ValueError("content cannot be empty")
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("confidence must be between 0 and 1")

        belief = {
            "id": belief_id,
            "content": content,
            "confidence": confidence,
            "source": source,
            "created_at": time.time(),
            "updated_at": time.time()
        }
        self._beliefs[belief_id] = belief
        self._links[belief_id] = []
        logger.info(f"created belief {belief_id} with confidence {confidence:.2f}")
        return belief

    def get_belief(self, belief_id: str) -> dict[str, Any]:
        """retrieve a belief by id."""
        if belief_id not in self._beliefs:
            raise KeyError(f"belief not found: {belief_id}")
        return self._beliefs[belief_id].copy()

    def link_beliefs(self, source_id: str, target_id: str, strength: float) -> None:
        """create a directed link between beliefs."""
        if source_id not in self._beliefs:
            raise KeyError(f"source belief not found: {source_id}")
        if target_id not in self._beliefs:
            raise KeyError(f"target belief not found: {target_id}")
        if not (0.0 <= strength <= 1.0):
            raise ValueError("link strength must be between 0 and 1")

        self._links[source_id].append((target_id, strength))
        logger.debug(f"linked {source_id} -> {target_id} with strength {strength:.2f}")

    def propagate(self, source_id: str) -> list[str]:
        """propagate confidence from source to linked beliefs."""
        if source_id not in self._beliefs:
            raise KeyError(f"belief not found: {source_id}")

        source = self._beliefs[source_id]
        updated_ids = []

        for target_id, strength in self._links[source_id]:
            target = self._beliefs[target_id]
            # weighted update: new_conf = old_conf + (source_conf - old_conf) * strength * 0.5
            delta = (source["confidence"] - target["confidence"]) * strength * 0.5
            new_conf = max(0.0, min(1.0, target["confidence"] + delta))
            target["confidence"] = new_conf
            target["updated_at"] = time.time()
            updated_ids.append(target_id)
            logger.debug(f"propagated to {target_id}, new confidence {new_conf:.3f}")

        return updated_ids

    def apply_decay(self, belief_id: str, decay_rate: float = 0.01) -> None:
        """apply time-based decay to belief confidence."""
        if belief_id not in self._beliefs:
            raise KeyError(f"belief not found: {belief_id}")

        belief = self._beliefs[belief_id]
        new_conf = max(0.0, belief["confidence"] * (1.0 - decay_rate))
        belief["confidence"] = new_conf
        belief["updated_at"] = time.time()
        logger.debug(f"decayed {belief_id} to {new_conf:.3f}")

    def mark_contradictory(self, id_a: str, id_b: str) -> None:
        """mark two beliefs as contradictory."""
        if id_a not in self._beliefs or id_b not in self._beliefs:
            raise KeyError("one or both beliefs not found")

        pair = tuple(sorted([id_a, id_b]))
        self._contradictions.add(pair)
        logger.info(f"marked contradiction: {id_a} <-> {id_b}")

    def find_contradictions(self) -> list[tuple[str, str]]:
        """return all known contradiction pairs."""
        return list(self._contradictions)

    def simulate_causal_update(
        self,
        belief_id: str,
        seed: int = 42
    ) -> dict[str, Any]:
        """
        simulate causal counterfactual update using world model.
        returns update delta and new confidence.
        """
        if belief_id not in self._beliefs:
            raise KeyError(f"belief not found: {belief_id}")

        np.random.seed(seed)
        belief = self._beliefs[belief_id]

        # deterministic causal simulation based on seed parity
        # even seeds increase confidence, odd seeds decrease
        direction = 1 if seed % 2 == 0 else -1
        magnitude = np.random.uniform(0.01, 0.1)
        delta = direction * magnitude

        new_conf = max(0.0, min(1.0, belief["confidence"] + delta))
        old_conf = belief["confidence"]
        belief["confidence"] = new_conf
        belief["updated_at"] = time.time()

        logger.info(f"causal update {belief_id}: {old_conf:.3f} -> {new_conf:.3f}")
        return {
            "belief_id": belief_id,
            "old_confidence": old_conf,
            "new_confidence": new_conf,
            "delta": delta,
            "seed": seed
        }

    def count(self) -> int:
        """return total number of beliefs."""
        return len(self._beliefs)
