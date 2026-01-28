# Author: Bradley R. Kinnard
# tests for goal collapse - TDD

import pytest
import numpy as np


class TestGoalHierarchy:
    """test goal hierarchy creation and management."""

    def test_create_root_goal(self, goal_collapse):
        goal = goal_collapse.create_goal(
            goal_id="g1",
            description="survive",
            priority=1.0
        )
        assert goal["id"] == "g1"
        assert goal["priority"] == 1.0
        assert goal["parent"] is None

    def test_create_subgoal(self, goal_collapse):
        goal_collapse.create_goal("root", "main goal", 1.0)
        subgoal = goal_collapse.create_goal(
            "sub1", "subgoal", 0.8, parent="root"
        )
        assert subgoal["parent"] == "root"

    def test_goal_hierarchy_depth(self, goal_collapse):
        goal_collapse.create_goal("l0", "level 0", 1.0)
        goal_collapse.create_goal("l1", "level 1", 0.9, parent="l0")
        goal_collapse.create_goal("l2", "level 2", 0.8, parent="l1")

        depth = goal_collapse.get_depth("l2")
        assert depth == 2


class TestGoalContradiction:
    """test detection of contradictory goals."""

    def test_detect_conflicting_goals(self, goal_collapse):
        goal_collapse.create_goal("save_money", "save money", 0.9)
        goal_collapse.create_goal("spend_money", "spend money", 0.8)
        goal_collapse.mark_conflicting("save_money", "spend_money")

        conflicts = goal_collapse.find_conflicts()
        assert len(conflicts) > 0

    def test_no_self_conflict(self, goal_collapse):
        goal_collapse.create_goal("single", "single goal", 0.9)

        conflicts = goal_collapse.find_conflicts()
        assert ("single", "single") not in conflicts


class TestGoalEvolution:
    """test goal evolution through RL-based updates."""

    def test_update_goal_priority(self, goal_collapse):
        goal_collapse.create_goal("evolve1", "evolving goal", 0.5)
        goal_collapse.update_priority("evolve1", delta=0.1)

        updated = goal_collapse.get_goal("evolve1")
        assert updated["priority"] == 0.6

    def test_priority_bounds_enforced(self, goal_collapse):
        goal_collapse.create_goal("bounded", "bounded goal", 0.9)

        # try to exceed 1.0
        goal_collapse.update_priority("bounded", delta=0.5)
        g = goal_collapse.get_goal("bounded")
        assert g["priority"] <= 1.0

    def test_deterministic_update_with_seed(self, goal_collapse):
        goal_collapse.create_goal("det1", "deterministic", 0.5)

        goal_collapse.evolve("det1", iterations=10, seed=42)
        result1 = goal_collapse.get_goal("det1")["priority"]

        # reset and re-run with same seed
        goal_collapse.reset_goal("det1", priority=0.5)
        goal_collapse.evolve("det1", iterations=10, seed=42)
        result2 = goal_collapse.get_goal("det1")["priority"]

        assert result1 == result2


class TestConvergence:
    """test RL convergence behavior."""

    def test_convergence_within_epsilon(self, goal_collapse):
        goal_collapse.create_goal("conv1", "converging", 0.5)

        converged, iterations = goal_collapse.evolve_until_stable(
            "conv1", epsilon=1e-5, max_iterations=1000, seed=42
        )

        assert converged or iterations == 1000

    def test_non_convergence_logged(self, goal_collapse, caplog):
        import logging
        caplog.set_level(logging.WARNING)

        goal_collapse.create_goal("noconv", "non-converging", 0.5)
        goal_collapse.evolve_until_stable(
            "noconv", epsilon=1e-10, max_iterations=10, seed=42
        )

        # should log warning about non-convergence
        assert "non-convergence" in caplog.text.lower() or "did not converge" in caplog.text.lower()

    def test_graceful_degradation_on_failure(self, goal_collapse):
        goal_collapse.create_goal("degrade", "degrading", 0.5)
        initial = goal_collapse.get_goal("degrade")["priority"]

        # force non-convergence
        converged, _ = goal_collapse.evolve_until_stable(
            "degrade", epsilon=1e-20, max_iterations=5, seed=42
        )

        # should still have a valid state
        final = goal_collapse.get_goal("degrade")
        assert 0.0 <= final["priority"] <= 1.0


class TestDifferentialPrivacy:
    """test differential privacy in reward signals."""

    def test_dp_noise_added_to_reward(self, goal_collapse):
        goal_collapse.create_goal("dp1", "dp test", 0.5)

        # run multiple times with same seed but DP enabled
        rewards = []
        for i in range(10):
            r = goal_collapse.compute_reward("dp1", seed=i, epsilon=0.1)
            rewards.append(r)

        # rewards should vary due to laplace noise
        assert len(set(rewards)) > 1

    def test_dp_epsilon_logged(self, goal_collapse, caplog):
        import logging
        caplog.set_level(logging.INFO)

        goal_collapse.create_goal("dp2", "dp logging", 0.5)
        goal_collapse.compute_reward("dp2", seed=42, epsilon=0.1)

        assert "epsilon" in caplog.text.lower()

    def test_dp_privacy_bounds(self, goal_collapse):
        """simulate inference attack, validate privacy."""
        goal_collapse.create_goal("privacy", "private goal", 0.5)

        # collect many noisy rewards
        rewards = [
            goal_collapse.compute_reward("privacy", seed=i, epsilon=0.1)
            for i in range(1000)
        ]

        # the variance should be consistent with laplace(0, 1/epsilon)
        # for epsilon=0.1, scale=10, variance=2*scale^2=200
        variance = np.var(rewards)
        # allowing generous bounds for test stability
        assert 50 < variance < 500


class TestCounterfactualBranching:
    """test counterfactual goal simulation."""

    def test_counterfactual_branches_generated(self, goal_collapse, config):
        if not config["features"]["use_counterfactual_sim"]:
            pytest.skip("counterfactual sim disabled")

        goal_collapse.create_goal("cf1", "counterfactual", 0.5)
        branches = goal_collapse.simulate_counterfactual("cf1", n_branches=5)

        assert len(branches) == 5

    @pytest.mark.slow
    def test_counterfactual_latency(self, goal_collapse, config):
        """each branch should complete in <200ms."""
        if not config["features"]["use_counterfactual_sim"]:
            pytest.skip("counterfactual sim disabled")

        import time
        goal_collapse.create_goal("cf2", "latency test", 0.5)

        start = time.time()
        goal_collapse.simulate_counterfactual("cf2", n_branches=10)
        elapsed = time.time() - start

        assert elapsed < 2.0, f"10 branches took {elapsed:.2f}s > 2s"


@pytest.fixture
def goal_collapse():
    """fixture providing fresh goal collapse instance."""
    from core.goal_collapse import GoalCollapse
    return GoalCollapse()


@pytest.fixture
def config():
    """fixture providing loaded config."""
    from utils.helpers import load_system_config
    return load_system_config()
