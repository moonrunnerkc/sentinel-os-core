# Author: Bradley R. Kinnard
# contradiction tracer - detects and resolves contradictory beliefs

import time
from typing import Any

from utils.helpers import get_logger


logger = get_logger(__name__)


# source priority ranking (higher = more trusted)
SOURCE_PRIORITY = {
    "sensor": 100,
    "observation": 90,
    "measurement": 85,
    "calculation": 70,
    "inference": 50,
    "assumption": 30,
    "default": 10
}


class ContradictionTracer:
    """
    tracks beliefs and detects/resolves contradictions.
    supports multiple resolution strategies.
    """

    def __init__(self):
        self._beliefs: dict[str, dict[str, Any]] = {}
        self._contradictions: set[tuple[str, str]] = set()

    def register_belief(
        self,
        belief_id: str,
        content: str,
        confidence: float = 0.5,
        source: str = "default",
        timestamp: float | None = None
    ) -> None:
        """register a belief for contradiction tracking."""
        self._beliefs[belief_id] = {
            "id": belief_id,
            "content": content,
            "confidence": confidence,
            "source": source,
            "timestamp": timestamp if timestamp is not None else time.time()
        }
        logger.debug(f"registered belief {belief_id}")

    def mark_contradictory(self, id_a: str, id_b: str) -> None:
        """mark two beliefs as contradictory."""
        if id_a not in self._beliefs or id_b not in self._beliefs:
            raise KeyError(f"belief not found: {id_a} or {id_b}")

        pair = tuple(sorted([id_a, id_b]))
        self._contradictions.add(pair)
        logger.info(f"marked contradiction between {id_a} and {id_b}")

    def detect_contradictions(self) -> list[tuple[str, str]]:
        """return all registered contradiction pairs."""
        result = list(self._contradictions)
        if result:
            logger.info(f"detected {len(result)} contradiction(s)")
        return result

    def resolve_by_confidence(
        self,
        id_a: str,
        id_b: str
    ) -> dict[str, str]:
        """resolve contradiction by keeping higher confidence belief."""
        a = self._beliefs.get(id_a)
        b = self._beliefs.get(id_b)
        if not a or not b:
            raise KeyError("belief not found")

        if a["confidence"] >= b["confidence"]:
            winner, loser = id_a, id_b
        else:
            winner, loser = id_b, id_a

        logger.info(f"resolved by confidence: winner={winner}, loser={loser}")
        return {"winner": winner, "loser": loser, "strategy": "confidence"}

    def resolve_by_recency(
        self,
        id_a: str,
        id_b: str
    ) -> dict[str, str]:
        """resolve contradiction by keeping more recent belief."""
        a = self._beliefs.get(id_a)
        b = self._beliefs.get(id_b)
        if not a or not b:
            raise KeyError("belief not found")

        if a["timestamp"] >= b["timestamp"]:
            winner, loser = id_a, id_b
        else:
            winner, loser = id_b, id_a

        logger.info(f"resolved by recency: winner={winner}, loser={loser}")
        return {"winner": winner, "loser": loser, "strategy": "recency"}

    def resolve_by_source(
        self,
        id_a: str,
        id_b: str
    ) -> dict[str, str]:
        """resolve contradiction by source priority."""
        a = self._beliefs.get(id_a)
        b = self._beliefs.get(id_b)
        if not a or not b:
            raise KeyError("belief not found")

        priority_a = SOURCE_PRIORITY.get(a["source"], SOURCE_PRIORITY["default"])
        priority_b = SOURCE_PRIORITY.get(b["source"], SOURCE_PRIORITY["default"])

        if priority_a >= priority_b:
            winner, loser = id_a, id_b
        else:
            winner, loser = id_b, id_a

        logger.info(f"resolved by source: winner={winner}, loser={loser}")
        return {"winner": winner, "loser": loser, "strategy": "source"}
