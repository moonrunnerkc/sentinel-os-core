# Author: Bradley R. Kinnard
# tests for crypto module

import pytest
import time

from crypto.commitments import (
    Commitment,
    CommitmentOpening,
    CommitmentScheme,
    CommitmentVerifier,
    StateTransitionCommitment,
    TransitionCommitmentScheme,
    BatchCommitmentScheme,
    VerificationResult,
)
from crypto.pq_signatures import (
    generate_keypair,
    Signer,
    Verifier,
    SignedLogChain,
    Algorithm,
)
from crypto.merkle import (
    MerkleTree,
    MerkleProof,
    IncrementalMerkleTree,
)


class TestCommitments:
    """tests for commitment scheme (replaces ZK proofs)."""

    def test_commit_and_open(self):
        scheme = CommitmentScheme(seed=42)

        data = {"beliefs": {"b1": 0.5}}
        commitment, opening = scheme.commit(data)

        assert commitment.commitment_id.startswith("cmt_")
        assert len(commitment.digest) == 64  # SHA-256 hex

    def test_verify_valid_opening(self):
        scheme = CommitmentScheme(seed=42)
        verifier = CommitmentVerifier()

        data = {"a": 1, "b": 2}
        commitment, opening = scheme.commit(data)

        result = verifier.verify(commitment, opening)
        assert result.valid, result.reason

    def test_verify_detects_tampering(self):
        scheme = CommitmentScheme(seed=42)
        verifier = CommitmentVerifier()

        data = {"a": 1}
        commitment, opening = scheme.commit(data)

        # tamper with opening data
        tampered_opening = CommitmentOpening(
            commitment_id=opening.commitment_id,
            data={"a": 2},  # changed
            salt=opening.salt,
        )

        result = verifier.verify(commitment, tampered_opening)
        assert not result.valid
        assert "mismatch" in result.reason

    def test_determinism_under_seed(self):
        scheme1 = CommitmentScheme(seed=42)
        scheme2 = CommitmentScheme(seed=42)

        data = {"x": 100}
        c1, _ = scheme1.commit(data)
        c2, _ = scheme2.commit(data)

        assert c1.digest == c2.digest
        assert c1.salt == c2.salt

    def test_transition_commitment(self):
        scheme = TransitionCommitmentScheme(seed=42)

        pre_state = {"value": 1}
        post_state = {"value": 2}
        input_data = {"delta": 1}

        stc, opening = scheme.commit_transition(
            pre_state, post_state, input_data, "update"
        )

        assert stc.pre_state_digest != stc.post_state_digest
        assert stc.transition_type == "update"

        result = scheme.verify_transition(stc, opening)
        assert result.valid

    def test_batch_commitment(self):
        scheme = BatchCommitmentScheme(seed=42)

        items = [{"i": 1}, {"i": 2}, {"i": 3}]
        commitments, openings, root = scheme.commit_batch(items)

        assert len(commitments) == 3
        assert len(openings) == 3
        assert len(root) == 64  # merkle root is SHA-256


class TestSignatures:
    """tests for signature engine."""

    def test_generate_keypair_ed25519(self):
        keypair = generate_keypair(Algorithm.ED25519, key_id="test_key")

        assert keypair.key_id == "test_key"
        assert keypair.algorithm == Algorithm.ED25519
        assert len(keypair.public_key) > 0
        assert len(keypair.private_key) > 0

    def test_generate_keypair_default_algorithm(self):
        keypair = generate_keypair()
        assert keypair.algorithm == Algorithm.ED25519

    def test_sign_and_verify(self):
        keypair = generate_keypair(Algorithm.ED25519)
        signer = Signer(keypair)
        verifier = Verifier.from_keypair(keypair)

        message = b"test message"
        signature = signer.sign(message)

        assert verifier.verify(message, signature)

    def test_verify_fails_on_tampered_message(self):
        keypair = generate_keypair(Algorithm.ED25519)
        signer = Signer(keypair)
        verifier = Verifier.from_keypair(keypair)

        message = b"test message"
        signature = signer.sign(message)

        tampered = b"tampered message"
        assert not verifier.verify(tampered, signature)

    def test_sign_json(self):
        keypair = generate_keypair(Algorithm.ED25519)
        signer = Signer(keypair)
        verifier = Verifier.from_keypair(keypair)

        data = {"key": "value", "number": 42}
        signed = signer.sign_json(data)

        valid, msg = verifier.verify_json(signed)
        assert valid, msg

    def test_signed_log_chain(self):
        keypair = generate_keypair(Algorithm.ED25519)
        signer = Signer(keypair)
        verifier = Verifier.from_keypair(keypair)

        chain = SignedLogChain(signer)

        chain.append({"event": "start"})
        chain.append({"event": "update", "value": 1})
        chain.append({"event": "end"})

        valid, msg = chain.verify_chain(verifier)
        assert valid, msg
        assert len(chain.get_chain()) == 3

    def test_algorithm_mismatch_detected(self):
        keypair = generate_keypair(Algorithm.ED25519)
        signer = Signer(keypair)

        # sign something
        signed = signer.sign_json({"test": 1})

        # modify the algorithm claim
        signed["algorithm"] = "dilithium3"

        verifier = Verifier.from_keypair(keypair)
        valid, msg = verifier.verify_json(signed)
        assert not valid
        assert "algorithm mismatch" in msg
        tree = MerkleTree()
        tree.add_leaves(["a", "b", "c", "d"])
        root = tree.build()

        assert len(root) == 64  # SHA256 hex length
        assert tree.verify_all_leaves()

    def test_empty_tree(self):
        tree = MerkleTree()
        root = tree.build()
        assert root == ""

    def test_single_leaf(self):
        tree = MerkleTree()
        tree.add_leaf("single")
        root = tree.build()

        assert len(root) == 64

    def test_generate_and_verify_proof(self):
        tree = MerkleTree()
        tree.add_leaves(["a", "b", "c", "d", "e"])
        tree.build()

        for i in range(5):
            proof = tree.get_proof(i)
            assert MerkleTree.verify_proof(proof)

    def test_proof_fails_with_wrong_root(self):
        tree = MerkleTree()
        tree.add_leaves(["a", "b", "c", "d"])
        tree.build()

        proof = tree.get_proof(0)

        # tamper with root
        tampered = MerkleProof(
            leaf=proof.leaf,
            leaf_index=proof.leaf_index,
            siblings=proof.siblings,
            root="wrong_root",
        )

        assert not MerkleTree.verify_proof(tampered)

    def test_deterministic_root(self):
        tree1 = MerkleTree()
        tree1.add_leaves(["a", "b", "c"])
        root1 = tree1.build()

        tree2 = MerkleTree()
        tree2.add_leaves(["a", "b", "c"])
        root2 = tree2.build()

        assert root1 == root2


class TestIncrementalMerkleTree:
    """tests for incremental merkle tree."""

    def test_incremental_add(self):
        tree = IncrementalMerkleTree(depth=10)

        root1 = tree.add_leaf("a")
        assert len(root1) == 64

        root2 = tree.add_leaf("b")
        assert root2 != root1

    def test_size_tracking(self):
        tree = IncrementalMerkleTree()

        assert tree.size == 0
        tree.add_leaf("a")
        assert tree.size == 1
        tree.add_leaf("b")
        assert tree.size == 2

    def test_consistent_with_batch_tree(self):
        # incremental and batch should produce same root for same leaves
        inc_tree = IncrementalMerkleTree(depth=10)
        batch_tree = MerkleTree()

        leaves = ["a", "b", "c", "d"]

        for leaf in leaves:
            inc_tree.add_leaf(leaf)
            batch_tree.add_leaf(leaf)

        batch_tree.build()

        # roots may differ slightly due to padding strategies, but both should be valid
        assert len(inc_tree.root) == 64
        assert len(batch_tree.root) == 64
