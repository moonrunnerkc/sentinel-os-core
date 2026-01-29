# Author: Bradley R. Kinnard
# tests for belief ecology - TDD: write tests first

import pytest
import time
from typing import Any


class TestBeliefCreation:
    """test belief creation and basic properties."""

    def test_create_belief_with_required_fields(self, belief_ecology):
        belief = belief_ecology.create_belief(
            belief_id="b001",
            content="the sky is blue",
            confidence=0.95,
            source="observation"
        )
        assert belief["id"] == "b001"
        assert belief["content"] == "the sky is blue"
        assert belief["confidence"] == 0.95
        assert belief["source"] == "observation"
        assert "created_at" in belief

    def test_create_belief_rejects_invalid_confidence(self, belief_ecology):
        with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
            belief_ecology.create_belief(
                belief_id="b002",
                content="test",
                confidence=1.5,
                source="test"
            )

    def test_create_belief_rejects_empty_content(self, belief_ecology):
        with pytest.raises(ValueError, match="content cannot be empty"):
            belief_ecology.create_belief(
                belief_id="b003",
                content="",
                confidence=0.5,
                source="test"
            )

    def test_create_belief_generates_timestamp(self, belief_ecology):
        belief = belief_ecology.create_belief(
            belief_id="b004",
            content="test belief",
            confidence=0.8,
            source="test"
        )
        assert isinstance(belief["created_at"], float)
        assert belief["created_at"] <= time.time()


class TestBeliefPropagation:
    """test belief propagation through the ecology."""

    def test_propagate_belief_updates_connected_beliefs(self, belief_ecology):
        # create parent and child beliefs
        parent = belief_ecology.create_belief("p1", "parent belief", 0.9, "test")
        child = belief_ecology.create_belief("c1", "child belief", 0.5, "test")
        belief_ecology.link_beliefs("p1", "c1", strength=0.8)

        # propagate from parent
        updated = belief_ecology.propagate("p1")

        # child confidence should be influenced by parent
        child_after = belief_ecology.get_belief("c1")
        assert child_after["confidence"] != 0.5

    def test_propagate_respects_link_strength(self, belief_ecology):
        parent = belief_ecology.create_belief("p2", "strong parent", 1.0, "test")
        weak_child = belief_ecology.create_belief("wc", "weak link", 0.5, "test")
        strong_child = belief_ecology.create_belief("sc", "strong link", 0.5, "test")

        belief_ecology.link_beliefs("p2", "wc", strength=0.1)
        belief_ecology.link_beliefs("p2", "sc", strength=0.9)

        belief_ecology.propagate("p2")

        wc = belief_ecology.get_belief("wc")
        sc = belief_ecology.get_belief("sc")

        # stronger link should have more influence
        assert abs(sc["confidence"] - 0.5) > abs(wc["confidence"] - 0.5)


class TestBeliefDecay:
    """test belief decay over time."""

    def test_decay_reduces_confidence(self, belief_ecology):
        belief = belief_ecology.create_belief("d1", "decaying belief", 0.9, "test")
        initial_conf = belief["confidence"]

        belief_ecology.apply_decay("d1", decay_rate=0.1)

        decayed = belief_ecology.get_belief("d1")
        assert decayed["confidence"] < initial_conf

    def test_decay_respects_minimum_threshold(self, belief_ecology):
        belief = belief_ecology.create_belief("d2", "low belief", 0.1, "test")

        # apply heavy decay
        for _ in range(10):
            belief_ecology.apply_decay("d2", decay_rate=0.5)

        final = belief_ecology.get_belief("d2")
        # should not go below minimum threshold
        assert final["confidence"] >= 0.0


class TestContradictionDetection:
    """test detection of contradictory beliefs."""

    def test_detect_direct_contradiction(self, belief_ecology):
        b1 = belief_ecology.create_belief("x1", "the cat is alive", 0.9, "test")
        b2 = belief_ecology.create_belief("x2", "the cat is dead", 0.9, "test")
        belief_ecology.mark_contradictory("x1", "x2")

        contradictions = belief_ecology.find_contradictions()
        assert ("x1", "x2") in contradictions or ("x2", "x1") in contradictions

    def test_no_false_positives(self, belief_ecology):
        b1 = belief_ecology.create_belief("y1", "the sky is blue", 0.9, "test")
        b2 = belief_ecology.create_belief("y2", "grass is green", 0.9, "test")

        contradictions = belief_ecology.find_contradictions()
        assert ("y1", "y2") not in contradictions
        assert ("y2", "y1") not in contradictions


class TestScalability:
    """test belief ecology at scale."""

    def test_small_scale_100_beliefs(self, belief_ecology):
        for i in range(100):
            belief_ecology.create_belief(f"s{i}", f"belief {i}", 0.5, "test")

        assert belief_ecology.count() == 100

    @pytest.mark.slow
    def test_large_scale_10k_beliefs(self, belief_ecology):
        start = time.time()
        for i in range(10000):
            belief_ecology.create_belief(f"l{i}", f"belief {i}", 0.5, "test")

        elapsed = time.time() - start
        assert belief_ecology.count() == 10000
        assert elapsed < 30, f"10k beliefs took {elapsed:.2f}s > 30s budget"


@pytest.fixture
def belief_ecology():
    """fixture providing a fresh belief ecology instance."""
    from core.belief_ecology import BeliefEcology
    return BeliefEcology()
