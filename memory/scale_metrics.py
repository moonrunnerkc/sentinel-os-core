# Author: Bradley R. Kinnard
# scale metrics - monitoring and reporting for scalability

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.helpers import get_logger

logger = get_logger(__name__)


@dataclass
class ScaleMetrics:
    """metrics for scalability monitoring."""
    belief_count: int = 0
    episode_count: int = 0
    goal_count: int = 0
    storage_bytes: int = 0
    index_size_bytes: int = 0
    memory_used_bytes: int = 0
    load_time_ms: float = 0.0
    query_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "belief_count": self.belief_count,
            "episode_count": self.episode_count,
            "goal_count": self.goal_count,
            "storage_bytes": self.storage_bytes,
            "index_size_bytes": self.index_size_bytes,
            "memory_used_bytes": self.memory_used_bytes,
            "load_time_ms": self.load_time_ms,
            "query_time_ms": self.query_time_ms,
            "timestamp": self.timestamp,
        }


class ScaleMonitor:
    """
    monitor system scale and warn when approaching limits.

    does NOT enforce hard limits - only warns and logs.
    hard limits are the caller's responsibility.
    """

    def __init__(
        self,
        soft_limit_warning_pct: float = 80.0,
        memory_limit_bytes: int | None = None,
    ):
        self._warning_pct = soft_limit_warning_pct
        self._memory_limit = memory_limit_bytes or self._get_available_memory()
        self._warnings_issued: set[str] = set()
        self._metrics_history: list[ScaleMetrics] = []

    def _get_available_memory(self) -> int:
        """get available system memory in bytes."""
        try:
            import psutil
            return psutil.virtual_memory().available
        except ImportError:
            # fallback: assume 8GB
            return 8 * 1024 * 1024 * 1024

    def _get_current_memory_usage(self) -> int:
        """get current process memory usage in bytes."""
        try:
            import psutil
            return psutil.Process().memory_info().rss
        except ImportError:
            # fallback: use sys.getsizeof for rough estimate
            return 0

    def check_memory_usage(self) -> tuple[float, bool]:
        """
        check memory usage percentage.

        returns (usage_pct, warning_issued)
        """
        current = self._get_current_memory_usage()
        pct = (current / self._memory_limit) * 100 if self._memory_limit > 0 else 0

        warning_issued = False
        if pct >= self._warning_pct and "memory" not in self._warnings_issued:
            logger.warning(
                f"memory usage at {pct:.1f}% of limit "
                f"({current / 1024 / 1024:.1f}MB / {self._memory_limit / 1024 / 1024:.1f}MB)"
            )
            self._warnings_issued.add("memory")
            warning_issued = True

        return pct, warning_issued

    def check_count_limit(
        self,
        name: str,
        current: int,
        limit: int | None,
    ) -> tuple[float, bool]:
        """
        check count against soft limit.

        returns (usage_pct, warning_issued)
        """
        if limit is None or limit <= 0:
            return 0.0, False

        pct = (current / limit) * 100
        warning_issued = False

        if pct >= self._warning_pct and name not in self._warnings_issued:
            logger.warning(f"{name} count at {pct:.1f}% of limit ({current} / {limit})")
            self._warnings_issued.add(name)
            warning_issued = True

        return pct, warning_issued

    def record_metrics(self, metrics: ScaleMetrics) -> None:
        """record metrics snapshot."""
        metrics.timestamp = time.time()
        self._metrics_history.append(metrics)

        # keep last 1000 snapshots
        if len(self._metrics_history) > 1000:
            self._metrics_history = self._metrics_history[-1000:]

    def get_current_metrics(
        self,
        belief_count: int = 0,
        episode_count: int = 0,
        goal_count: int = 0,
        storage_path: Path | None = None,
    ) -> ScaleMetrics:
        """build current metrics snapshot."""
        storage_bytes = 0
        if storage_path and storage_path.exists():
            if storage_path.is_file():
                storage_bytes = storage_path.stat().st_size
            elif storage_path.is_dir():
                storage_bytes = sum(f.stat().st_size for f in storage_path.rglob("*") if f.is_file())

        return ScaleMetrics(
            belief_count=belief_count,
            episode_count=episode_count,
            goal_count=goal_count,
            storage_bytes=storage_bytes,
            memory_used_bytes=self._get_current_memory_usage(),
        )

    def get_metrics_history(self) -> list[ScaleMetrics]:
        """return metrics history."""
        return self._metrics_history.copy()

    def reset_warnings(self) -> None:
        """reset warning state."""
        self._warnings_issued.clear()


class StorageQuotaExceeded(Exception):
    """raised when storage quota is exceeded."""
    pass


def estimate_memory_for_beliefs(count: int, avg_belief_size_bytes: int = 500) -> int:
    """estimate memory required for N beliefs."""
    # belief dict + index entry overhead
    overhead = 100  # bytes per belief for index
    return count * (avg_belief_size_bytes + overhead)


def estimate_storage_for_beliefs(count: int, avg_belief_size_bytes: int = 500) -> int:
    """estimate disk storage required for N beliefs."""
    # JSON overhead
    overhead = 50  # bytes per belief for JSON formatting
    return count * (avg_belief_size_bytes + overhead)
