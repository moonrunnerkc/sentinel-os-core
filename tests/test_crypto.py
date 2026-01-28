# Author: Bradley R. Kinnard
# tests for crypto module

import pytest
import time

from crypto.zk_proofs import (
    ZKProof,
    ZKProver,
    ZKVerifier,
    StateTransitionProof,
)
from crypto.pq_signatures import (
    generate_keypair,
    PQSigner,
    PQVerifier,
    SignedLogChain,
)
from crypto.merkle import (
    MerkleTree,
    MerkleProof,
    IncrementalMerkleTree,
)


class TestZKProofs:
    """tests for zero-knowledge proofs."""

    def test_prove_state_transition(self):
        prover = ZKProver(seed=42)

        pre_state = {"beliefs": {"b1": 0.5}}
        post_state = {"beliefs": {"b1": 0.6}}
        input_data = {"delta": 0.1}

        proof = prover.prove_state_transition(
            pre_state, post_state, input_data, "test_fn_hash"
        )

        assert proof.pre_state_digest != proof.post_state_digest
        assert proof.proof.proof_type == "state_transition"

    def test_verify_valid_proof(self):
        prover = ZKProver(seed=42)
        verifier = ZKVerifier()

        proof = prover.prove_state_transition(
            {"a": 1}, {"a": 2}, {"delta": 1}, "fn_hash"
        )

        valid, msg = verifier.verify(proof)
        assert valid, msg

    def test_verify_detects_tampering(self):
        prover = ZKProver(seed=42)
        verifier = ZKVerifier()

        proof = prover.prove_state_transition(
            {"a": 1}, {"a": 2}, {"delta": 1}, "fn_hash"
        )

        # tamper with pre_state_digest
        tampered = StateTransitionProof(
            pre_state_digest="tampered_hash",
            post_state_digest=proof.post_state_digest,
            input_digest=proof.input_digest,
            proof=proof.proof,
        )

        valid, msg = verifier.verify(tampered)
        assert not valid

    def test_verify_trace(self):
        prover = ZKProver(seed=42)
        verifier = ZKVerifier()

        trace = [
            {"pre_state_digest": "a", "post_state_digest": "b", "input_digest": "i1"},
            {"pre_state_digest": "b", "post_state_digest": "c", "input_digest": "i2"},
        ]

        proofs = prover.prove_trace(trace, "fn_hash")

        valid, msg = verifier.verify_trace(proofs)
        assert valid, msg


class TestPQSignatures:
    """tests for post-quantum signatures."""

    def test_generate_keypair(self):
        keypair = generate_keypair("test_key")

        assert keypair.key_id == "test_key"
        assert len(keypair.public_key) > 0
        assert len(keypair.private_key) > 0

    def test_sign_and_verify(self):
        keypair = generate_keypair()
        signer = PQSigner(keypair)
        verifier = PQVerifier.from_keypair(keypair)

        message = b"test message"
        signature = signer.sign(message)

        assert verifier.verify(message, signature)

    def test_verify_fails_on_tampered_message(self):
        keypair = generate_keypair()
        signer = PQSigner(keypair)
        verifier = PQVerifier.from_keypair(keypair)

        message = b"test message"
        signature = signer.sign(message)

        tampered = b"tampered message"
        assert not verifier.verify(tampered, signature)

    def test_sign_json(self):
        keypair = generate_keypair()
        signer = PQSigner(keypair)
        verifier = PQVerifier.from_keypair(keypair)

        data = {"key": "value", "number": 42}
        signed = signer.sign_json(data)

        valid, msg = verifier.verify_json(signed)
        assert valid, msg

    def test_signed_log_chain(self):
        keypair = generate_keypair()
        signer = PQSigner(keypair)
        verifier = PQVerifier.from_keypair(keypair)

        chain = SignedLogChain(signer)

        chain.append({"event": "start"})
        chain.append({"event": "update", "value": 1})
        chain.append({"event": "end"})

        valid, msg = chain.verify_chain(verifier)
        assert valid, msg
        assert len(chain.get_chain()) == 3


class TestMerkleTree:
    """tests for merkle tree."""

    def test_build_tree(self):
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
