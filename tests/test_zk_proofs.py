# Author: Bradley R. Kinnard
# tests for zk_proofs module - Pedersen commitments and Schnorr proofs

import pytest

from crypto.zk_proofs import (
    PedersenScheme,
    PedersenCommitment,
    SchnorrProver,
    SchnorrVerifier,
    StateTransitionProver,
    StateTransitionVerifier,
    MODULUS,
    GROUP_ORDER,
    G,
    P,
    Q,
    _mod_exp,
)


class TestPedersenCommitment:
    """tests for Pedersen commitment scheme."""

    def test_commit_creates_valid_commitment(self):
        scheme = PedersenScheme(seed=42)
        commitment = scheme.commit(100)

        assert commitment.commitment > 0
        assert commitment.value == 100
        assert commitment.blinding > 0

    def test_commit_different_values_different_commitments(self):
        scheme = PedersenScheme(seed=42)
        c1 = scheme.commit(100)
        c2 = scheme.commit(200)

        assert c1.commitment != c2.commitment

    def test_commit_same_value_same_seed_same_result(self):
        scheme1 = PedersenScheme(seed=42)
        scheme2 = PedersenScheme(seed=42)

        c1 = scheme1.commit(100)
        c2 = scheme2.commit(100)

        assert c1.commitment == c2.commitment
        assert c1.blinding == c2.blinding

    def test_verify_valid_opening(self):
        scheme = PedersenScheme(seed=42)
        commitment = scheme.commit(100)

        assert scheme.verify(
            commitment.commitment,
            commitment.value,
            commitment.blinding,
        )

    def test_verify_rejects_wrong_value(self):
        scheme = PedersenScheme(seed=42)
        commitment = scheme.commit(100)

        # try to open with wrong value
        assert not scheme.verify(
            commitment.commitment,
            101,  # wrong value
            commitment.blinding,
        )

    def test_verify_rejects_wrong_blinding(self):
        scheme = PedersenScheme(seed=42)
        commitment = scheme.commit(100)

        # try to open with wrong blinding
        assert not scheme.verify(
            commitment.commitment,
            commitment.value,
            commitment.blinding + 1,  # wrong blinding
        )

    def test_binding_cannot_find_collision(self):
        """test that we cannot find two different openings for same commitment."""
        scheme = PedersenScheme(seed=42)
        commitment = scheme.commit(100)

        # try many values - none should produce same commitment
        for v in range(1000):
            if v == 100:
                continue
            for r in range(10):
                if not scheme.verify(commitment.commitment, v, r):
                    pass  # expected
                else:
                    pytest.fail(f"found collision: v={v}, r={r}")

    def test_hiding_same_value_different_randomness(self):
        """test that same value with different randomness gives different commitment."""
        scheme = PedersenScheme()

        c1 = scheme.commit(100, blinding=123)
        c2 = scheme.commit(100, blinding=456)

        assert c1.commitment != c2.commitment
        assert c1.value == c2.value

    def test_homomorphic_addition(self):
        scheme = PedersenScheme(seed=42)

        c1 = scheme.commit(10)
        c2 = scheme.commit(20)
        c_sum = scheme.add(c1, c2)

        # verify the combined commitment opens to sum
        assert c_sum.value == 30
        assert scheme.verify(
            c_sum.commitment,
            c_sum.value,
            c_sum.blinding,
        )

    def test_negative_value_rejected(self):
        scheme = PedersenScheme(seed=42)

        with pytest.raises(ValueError, match="non-negative"):
            scheme.commit(-1)

    def test_public_property_hides_value(self):
        scheme = PedersenScheme(seed=42)
        commitment = scheme.commit(100)

        # public property should not reveal value
        public_data = commitment.to_dict()
        assert "value" not in public_data
        assert "blinding" not in public_data
        assert "commitment" in public_data


class TestSchnorrProof:
    """tests for Schnorr proof of knowledge."""

    def test_prove_and_verify_valid(self):
        prover = SchnorrProver(seed=42)
        verifier = SchnorrVerifier()

        secret = 12345
        public = _mod_exp(G, secret, P)

        proof = prover.prove(secret, public)
        valid, msg = verifier.verify(proof, public)

        assert valid, msg

    def test_verify_rejects_wrong_public(self):
        prover = SchnorrProver(seed=42)
        verifier = SchnorrVerifier()

        secret = 12345
        public = _mod_exp(G, secret, P)
        wrong_public = _mod_exp(G, secret + 1, P)

        proof = prover.prove(secret, public)
        valid, msg = verifier.verify(proof, wrong_public)

        assert not valid

    def test_soundness_random_proof_fails(self):
        """test that random bytes do not constitute valid proof."""
        from crypto.zk_proofs import SchnorrProof
        verifier = SchnorrVerifier()

        # random "proof"
        import random
        rng = random.Random(42)

        fake_proof = SchnorrProof(
            commitment_r=rng.randint(1, Q),
            challenge=rng.randint(1, Q),
            response=rng.randint(1, Q),
        )

        public = _mod_exp(G, 12345, P)
        valid, msg = verifier.verify(fake_proof, public)

        assert not valid

    def test_completeness_always_verifies_honest(self):
        """test that honest prover always produces valid proof."""
        prover = SchnorrProver(seed=42)
        verifier = SchnorrVerifier()

        # test with multiple secrets
        for secret in [1, 100, 10000, 999999]:
            public = _mod_exp(G, secret, P)
            proof = prover.prove(secret, public)
            valid, msg = verifier.verify(proof, public)

            assert valid, f"failed for secret={secret}: {msg}"

    def test_deterministic_under_seed(self):
        prover1 = SchnorrProver(seed=42)
        prover2 = SchnorrProver(seed=42)

        secret = 12345
        public = _mod_exp(G, secret, P)

        proof1 = prover1.prove(secret, public)
        proof2 = prover2.prove(secret, public)

        assert proof1.commitment_r == proof2.commitment_r
        assert proof1.challenge == proof2.challenge
        assert proof1.response == proof2.response


class TestStateTransitionProof:
    """tests for state transition proofs."""

    def test_belief_count_delta_valid(self):
        prover = StateTransitionProver(seed=42)

        proof, msg = prover.prove_belief_count_delta(
            pre_count=10,
            post_count=15,
            declared_delta=5,
        )

        assert proof is not None, msg
        assert proof.invariant_type == "BELIEF_COUNT"

    def test_belief_count_delta_invalid_rejected(self):
        prover = StateTransitionProver(seed=42)

        proof, msg = prover.prove_belief_count_delta(
            pre_count=10,
            post_count=15,
            declared_delta=3,  # wrong delta
        )

        assert proof is None
        assert "invariant violated" in msg

    def test_confidence_bound_valid(self):
        prover = StateTransitionProver(seed=42)

        proof, msg = prover.prove_confidence_bound(
            old_confidence=0.5,
            new_confidence=0.6,
            max_delta=0.2,
        )

        assert proof is not None, msg
        assert proof.invariant_type == "CONFIDENCE_BOUND"

    def test_confidence_bound_exceeded_rejected(self):
        prover = StateTransitionProver(seed=42)

        proof, msg = prover.prove_confidence_bound(
            old_confidence=0.5,
            new_confidence=0.9,  # delta = 0.4
            max_delta=0.2,       # max = 0.2
        )

        assert proof is None
        assert "invariant violated" in msg

    def test_verifier_accepts_valid_proof(self):
        prover = StateTransitionProver(seed=42)
        verifier = StateTransitionVerifier()

        proof, _ = prover.prove_belief_count_delta(10, 15, 5)
        valid, msg = verifier.verify(proof)

        assert valid, msg

    def test_verifier_rejects_zero_commitment(self):
        from crypto.zk_proofs import StateTransitionProof, SchnorrProof

        verifier = StateTransitionVerifier()

        # construct invalid proof with zero commitment
        bad_proof = StateTransitionProof(
            pre_state_commitment=123,
            post_state_commitment=456,
            delta_commitment=789,
            schnorr_proof=SchnorrProof(
                commitment_r=0,  # invalid
                challenge=123,
                response=456,
            ),
            invariant_type="BELIEF_COUNT",
            metadata={},
        )

        valid, msg = verifier.verify(bad_proof)
        assert not valid
        assert "zero commitment" in msg


class TestZKProofDeterminism:
    """tests for deterministic behavior under seed control."""

    def test_pedersen_deterministic(self):
        for seed in [1, 42, 999]:
            s1 = PedersenScheme(seed=seed)
            s2 = PedersenScheme(seed=seed)

            for val in [0, 1, 100, 999999]:
                c1 = s1.commit(val)
                c2 = s2.commit(val)
                assert c1.commitment == c2.commitment

    def test_schnorr_deterministic(self):
        for seed in [1, 42, 999]:
            p1 = SchnorrProver(seed=seed)
            p2 = SchnorrProver(seed=seed)

            for secret in [1, 100, 999]:
                public = _mod_exp(G, secret, P)
                proof1 = p1.prove(secret, public)
                proof2 = p2.prove(secret, public)
                assert proof1 == proof2

    def test_transition_proof_deterministic(self):
        for seed in [1, 42, 999]:
            p1 = StateTransitionProver(seed=seed)
            p2 = StateTransitionProver(seed=seed)

            proof1, _ = p1.prove_belief_count_delta(10, 20, 10)
            proof2, _ = p2.prove_belief_count_delta(10, 20, 10)

            assert proof1.pre_state_commitment == proof2.pre_state_commitment
            assert proof1.schnorr_proof == proof2.schnorr_proof
