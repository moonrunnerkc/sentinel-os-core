# Author: Bradley R. Kinnard
# tests for formal verification layer

import pytest
import tempfile
from pathlib import Path

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from verification.formal_checker import (
    Invariant,
    FormalChecker,
    InvariantViolation,
    CheckResult,
    VerificationReport,
    belief_confidence_bounded,
    goal_priority_non_negative,
    no_duplicate_belief_ids,
    episode_timestamps_ordered,
    state_hash_consistent,
    memory_within_limits,
    create_standard_checker,
)
from verification.proof_log import (
    ProofEntry,
    ProofLog,
    ProofLogger,
    run_formal_verification,
)


# strategies for property-based testing

belief_strategy = st.fixed_dictionaries({
    "id": st.text(min_size=1, max_size=20),
    "confidence": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    "content": st.text(max_size=100),
})

valid_state_strategy = st.fixed_dictionaries({
    "beliefs": st.dictionaries(
        st.text(min_size=1, max_size=10),
        st.fixed_dictionaries({
            "confidence": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        }),
        max_size=100,
    ),
    "goals": st.dictionaries(
        st.text(min_size=1, max_size=10),
        st.fixed_dictionaries({
            "priority": st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
        }),
        max_size=50,
    ),
    "episodes": st.lists(
        st.fixed_dictionaries({
            "timestamp": st.floats(min_value=0.0, max_value=1e12, allow_nan=False),
        }),
        max_size=100,
    ).map(lambda eps: sorted(eps, key=lambda e: e["timestamp"])),
})


class TestInvariantFunctions:
    """test individual invariant functions."""

    def test_belief_confidence_bounded_valid(self):
        state = {
            "beliefs": {
                "b1": {"confidence": 0.5},
                "b2": {"confidence": 0.0},
                "b3": {"confidence": 1.0},
            }
        }
        passed, msg = belief_confidence_bounded(state)
        assert passed

    def test_belief_confidence_bounded_invalid(self):
        state = {
            "beliefs": {
                "b1": {"confidence": 1.5},
            }
        }
        passed, msg = belief_confidence_bounded(state)
        assert not passed
        assert "b1" in msg

    def test_belief_confidence_negative(self):
        state = {
            "beliefs": {
                "b1": {"confidence": -0.1},
            }
        }
        passed, msg = belief_confidence_bounded(state)
        assert not passed

    def test_goal_priority_valid(self):
        state = {
            "goals": {
                "g1": {"priority": 0.0},
                "g2": {"priority": 5.0},
            }
        }
        passed, msg = goal_priority_non_negative(state)
        assert passed

    def test_goal_priority_invalid(self):
        state = {
            "goals": {
                "g1": {"priority": -1.0},
            }
        }
        passed, msg = goal_priority_non_negative(state)
        assert not passed
        assert "g1" in msg

    def test_no_duplicate_ids_valid(self):
        state = {
            "beliefs": {"b1": {}, "b2": {}, "b3": {}},
        }
        passed, msg = no_duplicate_belief_ids(state)
        assert passed

    def test_episode_timestamps_ordered_valid(self):
        state = {
            "episodes": [
                {"timestamp": 1.0},
                {"timestamp": 2.0},
                {"timestamp": 3.0},
            ]
        }
        passed, msg = episode_timestamps_ordered(state)
        assert passed

    def test_episode_timestamps_unordered(self):
        state = {
            "episodes": [
                {"timestamp": 3.0},
                {"timestamp": 1.0},
                {"timestamp": 2.0},
            ]
        }
        passed, msg = episode_timestamps_ordered(state)
        assert not passed

    def test_state_hash_consistent_no_hash(self):
        state = {"beliefs": {}}
        passed, msg = state_hash_consistent(state)
        assert passed  # skipped when no hash

    def test_memory_within_limits_no_limits(self):
        state = {
            "beliefs": {"b1": {}, "b2": {}},
            "episodes": [],
        }
        passed, msg = memory_within_limits(state)
        assert passed

    def test_memory_within_limits_exceeded(self):
        state = {
            "beliefs": {"b1": {}, "b2": {}, "b3": {}},
            "episodes": [],
            "_limits": {"max_beliefs": 2},
        }
        passed, msg = memory_within_limits(state)
        assert not passed


class TestFormalChecker:
    """test the formal checker."""

    def test_checker_creation(self):
        checker = FormalChecker()
        assert len(checker._invariants) == 0

    def test_register_invariant(self):
        checker = FormalChecker()
        inv = Invariant("test", lambda s: (True, "ok"))
        checker.register(inv)
        assert len(checker._invariants) == 1

    def test_register_fn(self):
        checker = FormalChecker()
        checker.register_fn("test", lambda s: (True, "ok"))
        assert len(checker._invariants) == 1

    def test_check_all_passes(self):
        checker = FormalChecker()
        checker.register_fn("always_pass", lambda s: (True, "ok"))

        report = checker.check_all({})
        assert report.all_passed
        assert report.passed == 1
        assert report.failed == 0

    def test_check_all_fails(self):
        checker = FormalChecker()
        checker.register_fn("always_fail", lambda s: (False, "bad"))

        report = checker.check_all({})
        assert not report.all_passed
        assert report.failed == 1

    def test_check_and_raise(self):
        checker = FormalChecker()
        checker.register_fn("always_fail", lambda s: (False, "bad"))

        with pytest.raises(InvariantViolation):
            checker.check_and_raise({})

    def test_fail_fast(self):
        checker = FormalChecker(fail_fast=True)
        checker.register_fn("fail1", lambda s: (False, "first"), critical=True)
        checker.register_fn("pass1", lambda s: (True, "ok"))

        report = checker.check_all({})
        # should stop after first critical failure
        assert report.total_checks == 1

    def test_no_fail_fast(self):
        checker = FormalChecker(fail_fast=False)
        checker.register_fn("fail1", lambda s: (False, "first"), critical=True)
        checker.register_fn("pass1", lambda s: (True, "ok"))

        report = checker.check_all({})
        # should check all
        assert report.total_checks == 2

    def test_history_tracking(self):
        checker = FormalChecker()
        checker.register_fn("test", lambda s: (True, "ok"))

        checker.check_all({})
        checker.check_all({})

        history = checker.get_history()
        assert len(history) == 2

    def test_standard_checker(self):
        checker = create_standard_checker()
        assert len(checker._invariants) == 6


class TestVerificationReport:
    """test verification report."""

    def test_report_digest_deterministic(self):
        report1 = VerificationReport(
            total_checks=1,
            passed=1,
            failed=0,
            skipped=0,
            results=[],
            duration_ms=1.0,
            timestamp=123.0,
        )
        report2 = VerificationReport(
            total_checks=1,
            passed=1,
            failed=0,
            skipped=0,
            results=[],
            duration_ms=1.0,
            timestamp=123.0,
        )
        assert report1.digest == report2.digest

    def test_report_to_dict(self):
        report = VerificationReport(
            total_checks=1,
            passed=1,
            failed=0,
            skipped=0,
            results=[],
            duration_ms=1.0,
            timestamp=123.0,
        )
        d = report.to_dict()
        assert "total_checks" in d
        assert "digest" in d


class TestProofLog:
    """test proof log."""

    def test_empty_log(self):
        log = ProofLog()
        assert len(log.entries) == 0

    def test_append_entry(self):
        log = ProofLog()
        entry = log.append(
            invariant_name="test",
            result="passed",
            message="ok",
            state_digest="abc123",
            duration_ms=1.0,
        )
        assert entry.sequence == 0
        assert len(log.entries) == 1

    def test_chain_verification(self):
        log = ProofLog()
        log.append("inv1", "passed", "ok", "abc", 1.0)
        log.append("inv2", "passed", "ok", "def", 1.0)

        valid, msg = log.verify_chain()
        assert valid

    def test_chain_tamper_detection(self):
        log = ProofLog()
        log.append("inv1", "passed", "ok", "abc", 1.0)

        # tamper with entry
        log.entries[0] = ProofEntry(
            sequence=0,
            invariant_name="tampered",
            result="passed",
            message="ok",
            state_digest="abc",
            timestamp=0,
            duration_ms=1.0,
        )

        valid, msg = log.verify_chain()
        assert not valid

    def test_save_and_load(self):
        log = ProofLog(seed=42)
        log.append("inv1", "passed", "ok", "abc", 1.0)
        log.append("inv2", "failed", "bad", "def", 2.0)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            log.save(path)
            loaded = ProofLog.load(path)

            assert len(loaded.entries) == 2
            assert loaded.seed == 42

            valid, msg = loaded.verify_chain()
            assert valid
        finally:
            path.unlink()

    def test_summary(self):
        log = ProofLog()
        log.append("inv1", "passed", "ok", "abc", 1.0)
        log.append("inv2", "failed", "bad", "def", 2.0)

        summary = log.summary()
        assert summary["total_checks"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1


class TestProofLogger:
    """test proof logger."""

    def test_state_digest(self):
        logger = ProofLogger()
        state = {"beliefs": {"b1": {}}}
        digest = logger.compute_state_digest(state)
        assert len(digest) == 16

    def test_state_digest_deterministic(self):
        logger = ProofLogger()
        state = {"beliefs": {"b1": {}}}
        d1 = logger.compute_state_digest(state)
        d2 = logger.compute_state_digest(state)
        assert d1 == d2

    def test_record_check(self):
        logger = ProofLogger()
        entry = logger.record_check(
            invariant_name="test",
            result="passed",
            message="ok",
            state={"beliefs": {}},
            duration_ms=1.0,
        )
        assert entry.sequence == 0

    def test_verify(self):
        logger = ProofLogger()
        logger.record_check("inv1", "passed", "ok", {}, 1.0)
        valid, msg = logger.verify()
        assert valid


class TestRunFormalVerification:
    """test the run_formal_verification function."""

    def test_valid_state(self):
        state = {
            "beliefs": {"b1": {"confidence": 0.5}},
            "goals": {"g1": {"priority": 1.0}},
            "episodes": [],
        }
        passed, log = run_formal_verification(state)
        assert passed
        assert len(log.entries) > 0

    def test_invalid_state(self):
        state = {
            "beliefs": {"b1": {"confidence": 1.5}},  # invalid
            "goals": {},
            "episodes": [],
        }
        passed, log = run_formal_verification(state)
        assert not passed

    def test_with_output_path(self):
        state = {
            "beliefs": {"b1": {"confidence": 0.5}},
            "goals": {},
            "episodes": [],
        }

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            passed, log = run_formal_verification(state, output_path=path)
            assert passed
            assert path.exists()

            # load and verify
            loaded = ProofLog.load(path)
            valid, msg = loaded.verify_chain()
            assert valid
        finally:
            path.unlink()


class TestPropertyBasedVerification:
    """property-based tests using Hypothesis."""

    @given(valid_state_strategy)
    @settings(max_examples=100)
    def test_valid_states_always_pass(self, state):
        """valid states should always pass verification."""
        passed, log = run_formal_verification(state)
        assert passed

    @given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    @settings(max_examples=50)
    def test_confidence_in_bounds(self, conf):
        """confidences in [0, 1] should always pass."""
        state = {"beliefs": {"b1": {"confidence": conf}}}
        passed, msg = belief_confidence_bounded(state)
        assert passed

    @given(st.floats(allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_confidence_out_of_bounds(self, conf):
        """confidences outside [0, 1] should fail."""
        assume(conf < 0 or conf > 1)
        state = {"beliefs": {"b1": {"confidence": conf}}}
        passed, msg = belief_confidence_bounded(state)
        assert not passed

    @given(st.floats(min_value=0.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_priority_non_negative(self, priority):
        """non-negative priorities should pass."""
        state = {"goals": {"g1": {"priority": priority}}}
        passed, msg = goal_priority_non_negative(state)
        assert passed

    @given(st.floats(max_value=-0.01, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_priority_negative_fails(self, priority):
        """negative priorities should fail."""
        state = {"goals": {"g1": {"priority": priority}}}
        passed, msg = goal_priority_non_negative(state)
        assert not passed

    @given(st.lists(st.floats(min_value=0, max_value=1e12, allow_nan=False), max_size=20))
    @settings(max_examples=50)
    def test_sorted_timestamps_pass(self, timestamps):
        """sorted timestamps should pass."""
        sorted_ts = sorted(timestamps)
        episodes = [{"timestamp": t} for t in sorted_ts]
        state = {"episodes": episodes}
        passed, msg = episode_timestamps_ordered(state)
        assert passed

    @given(st.lists(st.floats(min_value=0, max_value=1e12, allow_nan=False), min_size=3, max_size=20))
    @settings(max_examples=50)
    def test_unsorted_timestamps_may_fail(self, timestamps):
        """unsorted timestamps should fail if not in order."""
        sorted_ts = sorted(timestamps)
        if timestamps != sorted_ts:
            episodes = [{"timestamp": t} for t in timestamps]
            state = {"episodes": episodes}
            passed, msg = episode_timestamps_ordered(state)
            assert not passed


class TestViolationDetection:
    """test that violations are properly detected and logged."""

    def test_violation_raises_with_context(self):
        checker = FormalChecker()
        checker.register_fn("always_fail", lambda s: (False, "bad"))

        try:
            checker.check_and_raise({})
            assert False, "should have raised"
        except InvariantViolation as e:
            assert e.invariant_name == "always_fail"
            assert "bad" in str(e)

    def test_multiple_violations_all_logged(self):
        checker = FormalChecker(fail_fast=False)
        checker.register_fn("fail1", lambda s: (False, "first"), critical=False)
        checker.register_fn("fail2", lambda s: (False, "second"), critical=False)

        report = checker.check_all({})
        assert report.failed == 2

        failed_names = [r.name for r in report.results if r.result == CheckResult.FAILED]
        assert "fail1" in failed_names
        assert "fail2" in failed_names
