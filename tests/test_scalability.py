# Author: Bradley R. Kinnard
# tests for scalability - scale metrics, benchmarks, no hard caps

import pytest
import tempfile
from pathlib import Path

from memory.scale_metrics import (
    ScaleMetrics,
    ScaleMonitor,
    StorageQuotaExceeded,
    estimate_memory_for_beliefs,
    estimate_storage_for_beliefs,
)
from memory.persistent_memory import PersistentMemory
from memory.episodic_replay import EpisodicReplay


class TestScaleMetrics:
    """test scale metrics dataclass."""

    def test_metrics_creation(self):
        metrics = ScaleMetrics(
            belief_count=1000,
            episode_count=500,
            storage_bytes=1024 * 1024,
        )
        assert metrics.belief_count == 1000
        assert metrics.episode_count == 500
        assert metrics.storage_bytes == 1024 * 1024

    def test_metrics_to_dict(self):
        metrics = ScaleMetrics(belief_count=100)
        d = metrics.to_dict()
        assert d["belief_count"] == 100
        assert "timestamp" in d

    def test_metrics_defaults(self):
        metrics = ScaleMetrics()
        assert metrics.belief_count == 0
        assert metrics.episode_count == 0
        assert metrics.storage_bytes == 0


class TestScaleMonitor:
    """test scale monitoring."""

    def test_monitor_creation(self):
        monitor = ScaleMonitor(soft_limit_warning_pct=80)
        assert monitor._warning_pct == 80

    def test_check_count_limit_under(self):
        monitor = ScaleMonitor(soft_limit_warning_pct=80)
        pct, warned = monitor.check_count_limit("beliefs", 50, 100)
        assert pct == 50.0
        assert not warned

    def test_check_count_limit_at_warning(self):
        monitor = ScaleMonitor(soft_limit_warning_pct=80)
        pct, warned = monitor.check_count_limit("beliefs", 85, 100)
        assert pct == 85.0
        assert warned

    def test_check_count_limit_no_limit(self):
        monitor = ScaleMonitor()
        pct, warned = monitor.check_count_limit("beliefs", 1000000, None)
        assert pct == 0.0
        assert not warned

    def test_warning_only_issued_once(self):
        monitor = ScaleMonitor(soft_limit_warning_pct=80)
        _, warned1 = monitor.check_count_limit("beliefs", 85, 100)
        _, warned2 = monitor.check_count_limit("beliefs", 90, 100)
        assert warned1 is True
        assert warned2 is False  # already warned

    def test_reset_warnings(self):
        monitor = ScaleMonitor(soft_limit_warning_pct=80)
        monitor.check_count_limit("beliefs", 85, 100)
        monitor.reset_warnings()
        _, warned = monitor.check_count_limit("beliefs", 85, 100)
        assert warned is True

    def test_record_metrics(self):
        monitor = ScaleMonitor()
        metrics = ScaleMetrics(belief_count=100)
        monitor.record_metrics(metrics)
        history = monitor.get_metrics_history()
        assert len(history) == 1
        assert history[0].belief_count == 100

    def test_get_current_metrics(self):
        monitor = ScaleMonitor()
        metrics = monitor.get_current_metrics(belief_count=50, episode_count=25)
        assert metrics.belief_count == 50
        assert metrics.episode_count == 25


class TestEstimation:
    """test memory and storage estimation."""

    def test_estimate_memory(self):
        est = estimate_memory_for_beliefs(1000)
        assert est > 0
        # should be roughly 600KB for 1000 beliefs
        assert est < 1024 * 1024  # less than 1MB

    def test_estimate_storage(self):
        est = estimate_storage_for_beliefs(1000)
        assert est > 0
        assert est < 1024 * 1024  # less than 1MB

    def test_estimation_scales_linearly(self):
        est_1k = estimate_memory_for_beliefs(1000)
        est_10k = estimate_memory_for_beliefs(10000)
        assert abs(est_10k / est_1k - 10) < 0.1  # within 10%


class TestNoHardCaps:
    """test that hard caps are removed."""

    def test_persistent_memory_no_cap(self):
        memory = PersistentMemory()
        # should be able to store many beliefs without hitting a cap
        for i in range(1000):
            memory.store_belief({"id": f"belief_{i}", "content": f"test {i}"})
        assert len(memory.list_beliefs()) == 1000

    def test_episodic_replay_configurable_cap(self):
        # default is 10000, but can be set higher
        replay = EpisodicReplay(max_episodes=50000)
        assert replay._max_episodes == 50000


class TestScaleToLargeNumbers:
    """test scaling to large numbers of beliefs/episodes."""

    @pytest.mark.slow
    def test_scale_to_100k_beliefs(self):
        memory = PersistentMemory()
        for i in range(100000):
            memory.store_belief({"id": f"belief_{i}", "content": f"test {i}"})
        assert len(memory.list_beliefs()) == 100000

    def test_scale_to_10k_beliefs(self):
        memory = PersistentMemory()
        for i in range(10000):
            memory.store_belief({"id": f"belief_{i}", "content": f"test {i}"})
        assert len(memory.list_beliefs()) == 10000

    def test_scale_to_10k_episodes(self):
        replay = EpisodicReplay(max_episodes=15000)
        for i in range(10000):
            replay.record_episode({"id": f"episode_{i}", "data": f"test {i}"})
        assert replay.count() == 10000


class TestPersistence:
    """test persistence with large datasets."""

    def test_save_load_10k_beliefs(self):
        memory = PersistentMemory()
        for i in range(10000):
            memory.store_belief({"id": f"belief_{i}", "content": f"test {i}"})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "beliefs.json"
            memory.save_to_disk(path)

            memory2 = PersistentMemory()
            memory2.load_from_disk(path)

            assert len(memory2.list_beliefs()) == 10000

    def test_save_load_preserves_content(self):
        memory = PersistentMemory()
        for i in range(100):
            memory.store_belief({
                "id": f"belief_{i}",
                "content": f"content_{i}",
                "confidence": 0.5 + i / 200.0,
            })

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "beliefs.json"
            memory.save_to_disk(path)

            memory2 = PersistentMemory()
            memory2.load_from_disk(path)

            for i in range(100):
                belief = memory2.get_belief(f"belief_{i}")
                assert belief is not None
                assert belief["content"] == f"content_{i}"
