# Author: Bradley R. Kinnard
# tests for privacy module

import pytest
import numpy as np

from privacy.budget import (
    PrivacyBudget,
    PrivacyAccountant,
    BudgetExhaustedError,
)
from privacy.mechanisms import (
    laplace_mechanism,
    gaussian_mechanism,
    clip_l2_norm,
    randomized_response,
    exponential_mechanism,
    DPAggregator,
)


class TestPrivacyBudget:
    """tests for privacy budget tracking."""

    def test_initial_budget(self):
        budget = PrivacyBudget(total_epsilon=1.0, total_delta=1e-5)
        assert budget.remaining_epsilon() == 1.0
        assert budget.remaining_delta() == 1e-5
        assert not budget.is_exhausted()

    def test_can_spend(self):
        budget = PrivacyBudget(total_epsilon=1.0, total_delta=1e-5)
        assert budget.can_spend(0.5, 0.0)
        assert budget.can_spend(1.0, 1e-5)
        assert not budget.can_spend(1.1, 0.0)
        assert not budget.can_spend(0.5, 2e-5)

    def test_exhaustion(self):
        budget = PrivacyBudget(total_epsilon=1.0, total_delta=1e-5)
        budget.spent_epsilon = 1.0
        assert budget.is_exhausted()


class TestPrivacyAccountant:
    """tests for privacy accounting."""

    def test_spend_updates_budget(self):
        accountant = PrivacyAccountant(total_epsilon=1.0, total_delta=1e-5)

        accountant.spend(0.1, mechanism="laplace", operation="test")

        assert accountant.budget.spent_epsilon == 0.1
        assert accountant.budget.query_count == 1
        assert len(accountant.history) == 1

    def test_budget_exhaustion_raises(self):
        accountant = PrivacyAccountant(total_epsilon=0.5, total_delta=1e-5)

        with pytest.raises(BudgetExhaustedError):
            accountant.spend(0.6)

    def test_multiple_spends(self):
        accountant = PrivacyAccountant(total_epsilon=1.0)

        accountant.spend(0.3)
        accountant.spend(0.3)
        accountant.spend(0.3)

        assert abs(accountant.budget.spent_epsilon - 0.9) < 1e-10
        assert accountant.budget.query_count == 3

        with pytest.raises(BudgetExhaustedError):
            accountant.spend(0.2)

    def test_invariants_satisfied(self):
        accountant = PrivacyAccountant(total_epsilon=1.0)

        for _ in range(5):
            accountant.spend(0.1)

        valid, msg = accountant.check_invariants()
        assert valid, msg

    def test_audit_report(self):
        accountant = PrivacyAccountant(total_epsilon=1.0)
        accountant.spend(0.1, mechanism="laplace", operation="belief_update")

        report = accountant.export_audit_report()

        assert report["invariants_valid"]
        assert len(report["history"]) == 1
        assert report["budget"]["spent_epsilon"] == 0.1


class TestLaplaceMechanism:
    """tests for laplace mechanism."""

    def test_adds_noise(self):
        np.random.seed(42)
        value = 0.5
        noisy, meta = laplace_mechanism(value, sensitivity=1.0, epsilon=1.0)
        assert noisy != value
        assert meta["mechanism"] == "laplace"

    def test_deterministic_with_seed(self):
        v1, _ = laplace_mechanism(0.5, 1.0, 1.0, seed=42)
        v2, _ = laplace_mechanism(0.5, 1.0, 1.0, seed=42)
        assert v1 == v2

    def test_scale_inversely_proportional_to_epsilon(self):
        _, meta1 = laplace_mechanism(0.5, 1.0, 1.0, seed=42)
        _, meta2 = laplace_mechanism(0.5, 1.0, 0.1, seed=42)
        assert meta2["scale"] > meta1["scale"]

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError):
            laplace_mechanism(0.5, 1.0, 0.0)
        with pytest.raises(ValueError):
            laplace_mechanism(0.5, 1.0, -1.0)


class TestGaussianMechanism:
    """tests for gaussian mechanism."""

    def test_adds_noise(self):
        value = 0.5
        noisy, meta = gaussian_mechanism(value, 1.0, 1.0, 1e-5, seed=42)
        assert noisy != value
        assert meta["mechanism"] == "gaussian"

    def test_sigma_computation(self):
        _, meta = gaussian_mechanism(0.5, 1.0, 1.0, 1e-5, seed=42)
        # sigma should be > 0
        assert meta["sigma"] > 0


class TestClipL2Norm:
    """tests for gradient clipping."""

    def test_clips_large_vector(self):
        vector = np.array([3.0, 4.0])  # norm = 5
        clipped, meta = clip_l2_norm(vector, max_norm=1.0)

        assert np.linalg.norm(clipped) <= 1.0 + 1e-10
        assert meta["was_clipped"]

    def test_preserves_small_vector(self):
        vector = np.array([0.3, 0.4])  # norm = 0.5
        clipped, meta = clip_l2_norm(vector, max_norm=1.0)

        np.testing.assert_array_almost_equal(clipped, vector)
        assert not meta["was_clipped"]


class TestRandomizedResponse:
    """tests for randomized response mechanism."""

    def test_returns_boolean(self):
        response, meta = randomized_response(True, epsilon=1.0, seed=42)
        assert isinstance(response, (bool, np.bool_))

    def test_high_epsilon_mostly_truthful(self):
        np.random.seed(42)
        true_count = 0
        for i in range(100):
            response, _ = randomized_response(True, epsilon=10.0, seed=42 + i)
            if response:
                true_count += 1
        # with high epsilon, most responses should be truthful
        assert true_count > 90


class TestExponentialMechanism:
    """tests for exponential mechanism."""

    def test_selects_from_options(self):
        scores = {"a": 1.0, "b": 2.0, "c": 3.0}
        selected, meta = exponential_mechanism(scores, 1.0, 1.0, seed=42)
        assert selected in scores

    def test_higher_scores_more_likely(self):
        scores = {"low": 0.0, "high": 10.0}
        high_count = 0

        for i in range(100):
            selected, _ = exponential_mechanism(scores, 1.0, 1.0, seed=i)
            if selected == "high":
                high_count += 1

        assert high_count > 70  # high score should be selected more often


class TestDPAggregator:
    """tests for differentially private aggregation."""

    def test_aggregate_sum(self):
        aggregator = DPAggregator(epsilon=1.0, clip_bound=1.0)
        values = [0.5, 0.5, 0.5]

        noisy_sum, meta = aggregator.aggregate_sum(values, seed=42)

        # sum should be around 1.5 with some noise
        assert 0.0 < noisy_sum < 3.0

    def test_aggregate_mean(self):
        aggregator = DPAggregator(epsilon=1.0, clip_bound=1.0)
        values = [0.5, 0.5, 0.5]

        noisy_mean, meta = aggregator.aggregate_mean(values, seed=42)

        # mean should be around 0.5 with some noise
        assert 0.0 < noisy_mean < 1.0

    def test_clipping_applied(self):
        aggregator = DPAggregator(epsilon=1.0, clip_bound=0.5)
        values = [10.0, 10.0]  # will be clipped to 0.5 each

        noisy_sum, meta = aggregator.aggregate_sum(values, seed=42)

        # raw sum of clipped values is 1.0
        assert 0.0 < noisy_sum < 2.0
