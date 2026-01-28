# Author: Bradley R. Kinnard
# tests for contradiction tracer - TDD

import pytest


class TestContradictionDetection:
    """test contradiction detection logic."""

    def test_detect_explicit_contradiction(self, tracer):
        tracer.register_belief("a", "the door is open")
        tracer.register_belief("b", "the door is closed")
        tracer.mark_contradictory("a", "b")

        result = tracer.detect_contradictions()
        assert len(result) == 1
        assert {"a", "b"} == set(result[0])

    def test_no_contradiction_for_compatible_beliefs(self, tracer):
        tracer.register_belief("x", "the sky is blue")
        tracer.register_belief("y", "the grass is green")

        result = tracer.detect_contradictions()
        assert len(result) == 0

    def test_transitive_contradiction_detection(self, tracer):
        # a contradicts b, b contradicts c => potential chain
        tracer.register_belief("a", "state A")
        tracer.register_belief("b", "state B")
        tracer.register_belief("c", "state C")
        tracer.mark_contradictory("a", "b")
        tracer.mark_contradictory("b", "c")

        result = tracer.detect_contradictions()
        assert len(result) >= 2


class TestResolutionStrategies:
    """test contradiction resolution strategies."""

    def test_resolve_by_confidence(self, tracer):
        tracer.register_belief("high", "confident belief", confidence=0.95)
        tracer.register_belief("low", "uncertain belief", confidence=0.3)
        tracer.mark_contradictory("high", "low")

        resolved = tracer.resolve_by_confidence("high", "low")
        assert resolved["winner"] == "high"
        assert resolved["loser"] == "low"

    def test_resolve_by_recency(self, tracer):
        import time
        tracer.register_belief("old", "old belief", timestamp=1000.0)
        tracer.register_belief("new", "new belief", timestamp=2000.0)
        tracer.mark_contradictory("old", "new")

        resolved = tracer.resolve_by_recency("old", "new")
        assert resolved["winner"] == "new"

    def test_resolve_by_source_priority(self, tracer):
        tracer.register_belief("sensor", "sensor data", source="sensor")
        tracer.register_belief("inference", "inferred", source="inference")
        tracer.mark_contradictory("sensor", "inference")

        # sensor should win over inference
        resolved = tracer.resolve_by_source("sensor", "inference")
        assert resolved["winner"] == "sensor"


class TestContradictionLogging:
    """test that contradictions are properly logged."""

    def test_contradiction_logged_on_detection(self, tracer, caplog):
        import logging
        caplog.set_level(logging.INFO)

        tracer.register_belief("log1", "belief 1")
        tracer.register_belief("log2", "belief 2")
        tracer.mark_contradictory("log1", "log2")
        tracer.detect_contradictions()

        assert "contradiction" in caplog.text.lower()

    def test_resolution_logged(self, tracer, caplog):
        import logging
        caplog.set_level(logging.INFO)

        tracer.register_belief("r1", "belief 1", confidence=0.9)
        tracer.register_belief("r2", "belief 2", confidence=0.1)
        tracer.mark_contradictory("r1", "r2")
        tracer.resolve_by_confidence("r1", "r2")

        assert "resolved" in caplog.text.lower() or "winner" in caplog.text.lower()


@pytest.fixture
def tracer():
    """fixture providing a fresh contradiction tracer."""
    from core.contradiction_tracer import ContradictionTracer
    return ContradictionTracer()
