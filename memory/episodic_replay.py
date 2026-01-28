# Author: Bradley R. Kinnard
# episodic replay - LRU-based episode storage with eviction

import time
import json
from pathlib import Path
from typing import Any
from collections import OrderedDict

import numpy as np
import aiofiles

from utils.helpers import get_logger


logger = get_logger(__name__)


class EpisodicReplay:
    """
    episodic memory with LRU eviction policy.
    supports deterministic replay with seeding.
    """

    def __init__(self, max_episodes: int = 10000):
        self._max_episodes = max_episodes
        self._episodes: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._eviction_log: list[dict[str, Any]] = []

    def record_episode(self, episode: dict[str, Any]) -> None:
        """record a new episode with LRU eviction if at capacity."""
        if "id" not in episode:
            raise ValueError("episode must have 'id' field")

        episode_id = episode["id"]

        # check capacity and evict if needed
        while len(self._episodes) >= self._max_episodes:
            self._evict_oldest()

        # add with timestamp
        episode["recorded_at"] = time.time()
        self._episodes[episode_id] = episode
        self._episodes.move_to_end(episode_id)

        logger.debug(f"recorded episode {episode_id}")

    def _evict_oldest(self) -> None:
        """evict the least recently used episode."""
        if not self._episodes:
            return

        oldest_id, oldest_episode = self._episodes.popitem(last=False)

        self._eviction_log.append({
            "episode_id": oldest_id,
            "evicted_at": time.time(),
            "reason": "LRU capacity limit"
        })

        logger.info(f"evicted episode {oldest_id} (LRU)")

    def get_episode(self, episode_id: str) -> dict[str, Any] | None:
        """retrieve an episode, updating LRU order."""
        if episode_id not in self._episodes:
            return None

        self._episodes.move_to_end(episode_id)
        return self._episodes[episode_id].copy()

    def replay(
        self,
        n_episodes: int = 10,
        seed: int | None = None
    ) -> list[dict[str, Any]]:
        """sample episodes for replay, deterministic with seed."""
        if seed is not None:
            np.random.seed(seed)

        if not self._episodes:
            return []

        episode_ids = list(self._episodes.keys())
        n = min(n_episodes, len(episode_ids))

        if seed is not None:
            sampled_ids = list(np.random.choice(episode_ids, size=n, replace=False))
        else:
            sampled_ids = episode_ids[-n:]

        return [self._episodes[eid].copy() for eid in sampled_ids]

    def sample(
        self,
        n: int,
        seed: int | None = None
    ) -> list[dict[str, Any]]:
        """alias for replay with sampling."""
        return self.replay(n_episodes=n, seed=seed)

    def count(self) -> int:
        """return number of stored episodes."""
        return len(self._episodes)

    def get_eviction_log(self) -> list[dict[str, Any]]:
        """return eviction history."""
        return self._eviction_log.copy()

    def save_to_disk(self, path: Path | str) -> None:
        """save episodes to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "episodes": dict(self._episodes),
            "eviction_log": self._eviction_log
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"saved {len(self._episodes)} episodes to {path}")

    def load_from_disk(self, path: Path | str) -> None:
        """load episodes from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"file not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        self._episodes = OrderedDict(data.get("episodes", {}))
        self._eviction_log = data.get("eviction_log", [])

        logger.info(f"loaded {len(self._episodes)} episodes from {path}")

    async def async_save_to_disk(self, path: Path | str) -> None:
        """async save episodes to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "episodes": dict(self._episodes),
            "eviction_log": self._eviction_log
        }

        content = json.dumps(data, indent=2)
        async with aiofiles.open(path, "w") as f:
            await f.write(content)

        logger.info(f"async saved {len(self._episodes)} episodes to {path}")

    async def async_load_from_disk(self, path: Path | str) -> None:
        """async load episodes from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"file not found: {path}")

        async with aiofiles.open(path, "r") as f:
            content = await f.read()

        data = json.loads(content)
        self._episodes = OrderedDict(data.get("episodes", {}))
        self._eviction_log = data.get("eviction_log", [])

        logger.info(f"async loaded {len(self._episodes)} episodes from {path}")
