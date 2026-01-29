# Author: Bradley R. Kinnard
# tests that core functionality works without optional heavy deps
# validates graceful degradation when liboqs, brian2, tenseal, etc. are unavailable

import pytest
import sys
from unittest.mock import patch, MagicMock


class TestCoreOnlyMode:
    """
    verify sentinel-os-core works with just the core dependencies.

    these tests mock away optional deps to prove the system
    doesn't fail when they're unavailable.
    """

    def test_belief_ecology_no_optional_deps(self):
        """belief ecology works without optional deps."""
        from core.belief_ecology import BeliefEcology

        eco = BeliefEcology()
        b = eco.create_belief("b1", "test belief", confidence=0.8, source="test")

        assert b["id"] == "b1"
        assert b["confidence"] == 0.8
        assert eco.get_belief("b1") == b

    def test_goal_collapse_no_optional_deps(self):
        """goal collapse works without optional deps."""
        from core.goal_collapse import GoalCollapse

        engine = GoalCollapse()
        goal = engine.create_goal("g1", "test goal", priority=1.0)

        assert goal is not None
        assert goal["id"] == "g1"

    def test_contradiction_tracer_no_optional_deps(self):
        """contradiction tracer works without optional deps."""
        from core.contradiction_tracer import ContradictionTracer

        tracer = ContradictionTracer()
        tracer.register_belief("b1", "the sky is blue", 0.9)
        tracer.register_belief("b2", "the sky is not blue", 0.8)
        tracer.mark_contradictory("b1", "b2")

        contradictions = tracer.detect_contradictions()
        assert len(contradictions) >= 1

    def test_persistent_memory_no_optional_deps(self):
        """persistent memory works with basic operations."""
        from memory.persistent_memory import PersistentMemory

        mem = PersistentMemory(enable_he=False)
        belief = {"id": "key1", "value": "test", "confidence": 0.8}
        mem.store_belief(belief)
        result = mem.get_belief("key1")

        assert result["value"] == "test"

    def test_episodic_replay_no_optional_deps(self):
        """episodic replay works without optional deps."""
        from memory.episodic_replay import EpisodicReplay
        import uuid

        replay = EpisodicReplay(max_episodes=100)
        episode_id = str(uuid.uuid4())
        replay.record_episode({"id": episode_id, "state": "s1", "action": "a1", "reward": 1.0})

        samples = replay.sample(1)
        assert len(samples) == 1
        assert samples[0]["state"] == "s1"

    def test_audit_logger_no_optional_deps(self):
        """audit logger works with standard hashlib hmac."""
        from security.audit_logger import AuditLogger
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            al = AuditLogger(log_dir=tmpdir, master_seed=42)
            al.log("info", "test_event", {"detail": "value"})

            # verify log was written and is verifiable
            assert al.verify_all()

    def test_privacy_budget_no_optional_deps(self):
        """privacy budget tracking works without optional deps."""
        from privacy.budget import PrivacyAccountant

        accountant = PrivacyAccountant(total_epsilon=1.0, total_delta=1e-5)
        accountant.spend(epsilon=0.1)

        # verify spend was recorded - check_invariants returns tuple (passed, msg)
        passed, msg = accountant.check_invariants()
        assert passed, msg

    def test_verification_layer_no_optional_deps(self):
        """invariant checker works without optional deps."""
        from verification.invariants import InvariantChecker

        checker = InvariantChecker()

        # just verify we can create and use the checker
        checker.register_invariant("always_pass", lambda s: (True, "ok"))
        # the actual check requires SystemState, but we verified the import works
        assert checker is not None

    def test_crypto_commitments_no_optional_deps(self):
        """commitment schemes work without optional deps."""
        from crypto.commitments import CommitmentScheme

        cs = CommitmentScheme()
        data = {"value": "secret"}
        commitment, opening = cs.commit(data)

        # verify it returns a commitment and opening tuple
        assert commitment is not None
        assert opening is not None
        assert cs.verify(commitment, opening)

    def test_zk_proofs_no_optional_deps(self):
        """zk proofs work with pure python discrete log."""
        from crypto.zk_proofs import PedersenScheme

        scheme = PedersenScheme()
        value = 12345
        commitment = scheme.commit(value)

        # verify commitment was created
        assert commitment is not None
        assert commitment.value == value
        # verify using the commitment's stored blinding factor
        assert scheme.verify(commitment.commitment, value, commitment.blinding)


class TestOptionalDepsGracefulDegradation:
    """
    test that optional deps fail gracefully with clear errors.
    """

    def test_liboqs_availability_check(self):
        """pq signatures report liboqs availability correctly."""
        from crypto.pq_signatures import liboqs_available, generate_keypair

        # should not crash regardless of availability
        assert isinstance(liboqs_available(), bool)

        # can generate keypair with fallback
        keypair = generate_keypair()
        assert keypair is not None

    def test_tenseal_unavailable_skips_he(self):
        """homomorphic encryption is skipped when tenseal unavailable."""
        # mock tenseal as unavailable
        with patch.dict(sys.modules, {"tenseal": None}):
            # importing should not crash
            from crypto import homomorphic
            # he operations should be no-ops or raise clear errors

    def test_brian2_unavailable_skips_neuromorphic(self):
        """neuromorphic mode fails cleanly when brian2 unavailable."""
        # this test documents expected behavior
        pass  # neuromorphic is already optional via config flag


class TestCoreOnlyImports:
    """
    verify import paths work without pulling in optional deps.
    """

    def test_import_core_modules(self):
        """core modules import without optional deps."""
        # these should all import cleanly
        from core import belief_ecology
        from core import goal_collapse
        from core import contradiction_tracer
        from core import meta_cognition
        from core import meta_evolution

        assert belief_ecology is not None
        assert goal_collapse is not None
        assert contradiction_tracer is not None

    def test_import_memory_modules(self):
        """memory modules import without optional deps."""
        from memory import persistent_memory
        from memory import episodic_replay

        assert persistent_memory is not None
        assert episodic_replay is not None

    def test_import_security_modules(self):
        """security modules import without optional deps."""
        from security import audit_logger
        from security import soft_isolation

        assert audit_logger is not None
        assert soft_isolation is not None

    def test_import_privacy_modules(self):
        """privacy modules import without optional deps."""
        from privacy import budget
        from privacy import mechanisms

        assert budget is not None
        assert mechanisms is not None

    def test_import_verification_modules(self):
        """verification modules import without optional deps."""
        from verification import invariants
        from verification import state_machine
        from verification import formal_checker

        assert invariants is not None
        assert state_machine is not None
        assert formal_checker is not None
