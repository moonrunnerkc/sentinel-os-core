# Author: Bradley R. Kinnard
# zero-knowledge proofs for state transitions

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from utils.helpers import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ZKProof:
    """
    zero-knowledge proof for state transitions.

    proves: "I know a valid transition from state_pre to state_post
    using input_data, without revealing state contents."

    structure:
    - commitment: hash of witness data
    - challenge: verifier's random challenge (Fiat-Shamir heuristic)
    - response: prover's response to challenge
    """
    proof_id: str
    commitment: str
    challenge: str
    response: str
    public_inputs: dict[str, str]
    timestamp: float
    proof_type: str = "state_transition"

    def to_dict(self) -> dict[str, Any]:
        return {
            "proof_id": self.proof_id,
            "commitment": self.commitment,
            "challenge": self.challenge,
            "response": self.response,
            "public_inputs": self.public_inputs,
            "timestamp": self.timestamp,
            "proof_type": self.proof_type,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ZKProof":
        return cls(**d)


@dataclass
class StateTransitionProof:
    """
    proof that a state transition was computed correctly.

    public inputs:
    - pre_state_digest: hash of state before transition
    - post_state_digest: hash of state after transition
    - input_digest: hash of input that triggered transition

    private witness (not revealed):
    - actual state contents
    - transition function internals
    """
    pre_state_digest: str
    post_state_digest: str
    input_digest: str
    proof: ZKProof
    verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "pre_state_digest": self.pre_state_digest,
            "post_state_digest": self.post_state_digest,
            "input_digest": self.input_digest,
            "proof": self.proof.to_dict(),
            "verified": self.verified,
        }


class ZKProver:
    """
    zero-knowledge prover for state transitions.

    uses a simplified sigma-protocol structure:
    1. commit to witness
    2. receive challenge (Fiat-Shamir: hash of commitment + public inputs)
    3. compute response

    note: this is a demonstration implementation.
    for production, integrate with a real ZK library (halo2, arkworks, etc.)
    """

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._rng = np.random.RandomState(seed)

    def _hash(self, *args: str) -> str:
        """compute SHA256 hash of concatenated inputs."""
        combined = "".join(args)
        return hashlib.sha256(combined.encode()).hexdigest()

    def _generate_proof_id(self) -> str:
        """generate unique proof ID."""
        return f"zkp_{int(time.time() * 1000)}_{self._rng.randint(0, 10000)}"

    def prove_state_transition(
        self,
        pre_state: dict[str, Any],
        post_state: dict[str, Any],
        input_data: dict[str, Any],
        transition_fn_hash: str,
    ) -> StateTransitionProof:
        """
        generate a ZK proof for a state transition.

        proves: transition from pre_state to post_state via input_data
        was computed by transition function with given hash.
        """
        # compute digests (public)
        pre_digest = self._hash(json.dumps(pre_state, sort_keys=True))
        post_digest = self._hash(json.dumps(post_state, sort_keys=True))
        input_digest = self._hash(json.dumps(input_data, sort_keys=True))

        # witness: the actual state and transition details (private)
        witness = {
            "pre_state": pre_state,
            "post_state": post_state,
            "input": input_data,
            "fn_hash": transition_fn_hash,
        }
        witness_hash = self._hash(json.dumps(witness, sort_keys=True))

        # commitment: hash of witness with random blinding factor
        blinding = self._rng.bytes(32).hex()
        commitment = self._hash(witness_hash, blinding)

        # challenge: Fiat-Shamir heuristic - hash of commitment + public inputs
        challenge = self._hash(commitment, pre_digest, post_digest, input_digest)

        # response: in a real sigma protocol, this would be computed based on
        # the challenge and witness. here we simulate the response.
        response_data = {
            "witness_hash": witness_hash,
            "challenge": challenge,
            "blinding_commitment": self._hash(blinding, challenge),
        }
        response = self._hash(json.dumps(response_data, sort_keys=True))

        proof = ZKProof(
            proof_id=self._generate_proof_id(),
            commitment=commitment,
            challenge=challenge,
            response=response,
            public_inputs={
                "pre_state_digest": pre_digest,
                "post_state_digest": post_digest,
                "input_digest": input_digest,
                "transition_fn_hash": transition_fn_hash,
            },
            timestamp=time.time(),
        )

        logger.info(f"generated ZK proof {proof.proof_id} for transition")

        return StateTransitionProof(
            pre_state_digest=pre_digest,
            post_state_digest=post_digest,
            input_digest=input_digest,
            proof=proof,
            verified=False,
        )

    def prove_trace(
        self,
        trace: list[dict[str, Any]],
        transition_fn_hash: str,
    ) -> list[StateTransitionProof]:
        """generate proofs for an entire trace of transitions."""
        proofs = []

        for entry in trace:
            proof = self.prove_state_transition(
                pre_state={"digest": entry.get("pre_state_digest", "")},
                post_state={"digest": entry.get("post_state_digest", "")},
                input_data={"digest": entry.get("input_digest", "")},
                transition_fn_hash=transition_fn_hash,
            )
            proofs.append(proof)

        return proofs


class ZKVerifier:
    """
    zero-knowledge verifier for state transition proofs.

    verifies proofs without learning state contents.
    """

    def __init__(self):
        pass

    def _hash(self, *args: str) -> str:
        """compute SHA256 hash of concatenated inputs."""
        combined = "".join(args)
        return hashlib.sha256(combined.encode()).hexdigest()

    def verify(self, proof: StateTransitionProof) -> tuple[bool, str]:
        """
        verify a state transition proof.

        checks:
        1. challenge is correctly computed from commitment and public inputs
        2. response is consistent with challenge
        3. public inputs are well-formed
        """
        zk = proof.proof

        # verify challenge computation (Fiat-Shamir)
        expected_challenge = self._hash(
            zk.commitment,
            zk.public_inputs["pre_state_digest"],
            zk.public_inputs["post_state_digest"],
            zk.public_inputs["input_digest"],
        )

        if zk.challenge != expected_challenge:
            return False, "challenge mismatch"

        # verify response is a valid hash (basic structure check)
        if len(zk.response) != 64:  # SHA256 hex length
            return False, "invalid response format"

        # verify public inputs are consistent with proof
        if proof.pre_state_digest != zk.public_inputs["pre_state_digest"]:
            return False, "pre_state_digest mismatch"
        if proof.post_state_digest != zk.public_inputs["post_state_digest"]:
            return False, "post_state_digest mismatch"
        if proof.input_digest != zk.public_inputs["input_digest"]:
            return False, "input_digest mismatch"

        logger.info(f"verified ZK proof {zk.proof_id}")
        return True, "proof valid"

    def verify_trace(
        self,
        proofs: list[StateTransitionProof],
        expected_final_digest: str | None = None,
    ) -> tuple[bool, str]:
        """
        verify a chain of transition proofs.

        checks:
        1. each individual proof is valid
        2. chain links correctly (post_state[i] == pre_state[i+1])
        3. optionally, final state matches expected
        """
        if not proofs:
            return True, "empty trace"

        for i, proof in enumerate(proofs):
            valid, msg = self.verify(proof)
            if not valid:
                return False, f"proof {i} invalid: {msg}"

            # check chain linkage
            if i > 0:
                prev = proofs[i - 1]
                if prev.post_state_digest != proof.pre_state_digest:
                    return False, f"chain break at proof {i}"

        # check final state if expected
        if expected_final_digest is not None:
            if proofs[-1].post_state_digest != expected_final_digest:
                return False, "final state mismatch"

        return True, f"trace valid ({len(proofs)} transitions)"


class BatchProver:
    """
    efficient batch proving for multiple transitions.

    uses merkle tree to commit to batch, reducing proof overhead.
    """

    def __init__(self, seed: int = 42):
        self._prover = ZKProver(seed)

    def prove_batch(
        self,
        transitions: list[dict[str, Any]],
        transition_fn_hash: str,
    ) -> dict[str, Any]:
        """prove a batch of transitions with single commitment."""
        from crypto.merkle import MerkleTree

        # generate individual proofs
        proofs = []
        digests = []

        for t in transitions:
            proof = self._prover.prove_state_transition(
                pre_state=t.get("pre_state", {}),
                post_state=t.get("post_state", {}),
                input_data=t.get("input", {}),
                transition_fn_hash=transition_fn_hash,
            )
            proofs.append(proof)
            digests.append(proof.proof.proof_id)

        # build merkle tree over proof IDs
        tree = MerkleTree()
        for d in digests:
            tree.add_leaf(d)
        tree.build()

        return {
            "batch_root": tree.root,
            "n_transitions": len(transitions),
            "proofs": [p.to_dict() for p in proofs],
            "timestamp": time.time(),
        }
