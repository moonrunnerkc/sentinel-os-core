# Author: Bradley R. Kinnard
# tests for smt-based formal verification via z3

import pytest
from verification.smt_checker import SMTChecker, SMTResult, Z3_AVAILABLE


@pytest.mark.skipif(not Z3_AVAILABLE, reason="z3-solver not installed")
class TestSMTCheckerWithZ3:
    """tests that require z3 to be installed."""

    def test_confidence_bounded_proved(self):
        """verify confidence bounds are provable."""
        checker = SMTChecker()
        result = checker.check_confidence_bounded()

        assert result.proved
        assert result.counterexample is None
        assert "proved" in result.message.lower()

    def test_priority_non_negative_proved(self):
        """verify priority non-negativity is provable."""
        checker = SMTChecker()
        result = checker.check_priority_non_negative()

        assert result.proved
        assert result.counterexample is None

    def test_decay_monotonic_proved(self):
        """verify decay monotonicity is provable."""
        checker = SMTChecker()
        result = checker.check_decay_monotonic()

        assert result.proved
        assert result.counterexample is None

    def test_budget_exhaustion_proved(self):
        """verify budget non-negativity is provable."""
        checker = SMTChecker()
        result = checker.check_budget_exhaustion(epsilon_total=1.0)

        assert result.proved
        assert result.counterexample is None

    def test_run_all_checks(self):
        """verify all checks run and pass."""
        checker = SMTChecker()
        results = checker.run_all_checks()

        assert len(results) == 4
        assert all(r.proved for r in results)

    def test_smt_result_to_dict(self):
        """verify smt result serialization."""
        result = SMTResult(
            property_name="test",
            satisfiable=False,
            counterexample=None,
            proved=True,
            message="proved",
        )
        d = result.to_dict()

        assert d["property_name"] == "test"
        assert d["proved"] is True


class TestSMTCheckerGracefulDegradation:
    """tests that work regardless of z3 availability."""

    def test_checker_available_property(self):
        """verify available property reflects z3 status."""
        checker = SMTChecker()
        assert checker.available == Z3_AVAILABLE

    def test_graceful_skip_when_unavailable(self):
        """verify checks return skip message when z3 unavailable."""
        if Z3_AVAILABLE:
            pytest.skip("z3 is available, testing skip behavior not applicable")

        checker = SMTChecker()
        result = checker.check_confidence_bounded()

        assert not result.proved
        assert "skipped" in result.message.lower() or "not available" in result.message.lower()


class TestSMTCheckerMocked:
    """tests with mocked z3 for CI without z3 installed."""

    def test_result_dataclass(self):
        """verify SMTResult works without z3."""
        result = SMTResult(
            property_name="test_prop",
            satisfiable=True,
            counterexample={"x": 0.5},
            proved=False,
            message="found counterexample",
        )

        assert result.property_name == "test_prop"
        assert result.satisfiable
        assert result.counterexample == {"x": 0.5}
        assert not result.proved

    def test_result_serialization(self):
        """verify SMTResult serializes properly."""
        result = SMTResult(
            property_name="test",
            satisfiable=False,
            counterexample=None,
            proved=True,
            message="ok",
        )

        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["property_name"] == "test"
        assert d["proved"] is True
