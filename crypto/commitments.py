# Author: Bradley R. Kinnard
# cryptographic commitment scheme for state transitions
# replaces previous "zk_proofs" module - this is NOT zero-knowledge

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from utils.helpers import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Commitment:
    """
    cryptographic commitment to data.

    properties:
    - binding: cannot open to different data
    - hiding: commitment reveals nothing without opening (salt provides this)

    NOT zero-knowledge: verifier learns data when opening is revealed.
    """
    commitment_id: str
    digest: str              # SHA-256(canonical(data) + salt)
    salt: str                # 32-byte random hex
    public_metadata: dict    # non-hidden fields
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "commitment_id": self.commitment_id,
            "digest": self.digest,
            "salt": self.salt,
            "public_metadata": self.public_metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Commitment":
        return cls(
            commitment_id=d["commitment_id"],
            digest=d["digest"],
            salt=d["salt"],
            public_metadata=d.get("public_metadata", {}),
            created_at=d["created_at"],
        )


@dataclass(frozen=True)
class CommitmentOpening:
    """
    opening for a commitment - reveals the committed data.
    """
    commitment_id: str
    data: dict
    salt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "commitment_id": self.commitment_id,
            "data": self.data,
            "salt": self.salt,
        }


@dataclass(frozen=True)
class VerificationResult:
    """result of commitment verification."""
    valid: bool
    reason: str
    commitment_id: str


class CommitmentScheme:
    """
    hash-based commitment scheme with salted SHA-256.

    usage:
    1. commit(data) -> Commitment (share publicly)
    2. later: open(commitment, data) -> CommitmentOpening (reveal)
    3. verify(commitment, opening) -> VerificationResult
    """

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._counter = 0

    def _canonical(self, data: dict) -> str:
        """produce canonical JSON representation."""
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    def _hash(self, data: str) -> str:
        """SHA-256 hash."""
        return hashlib.sha256(data.encode()).hexdigest()

    def _generate_salt(self) -> str:
        """generate 32-byte random salt (deterministic under seed)."""
        return self._rng.bytes(32).hex()

    def _generate_id(self) -> str:
        """generate unique commitment ID."""
        self._counter += 1
        return f"cmt_{int(time.time() * 1000)}_{self._counter}"

    def commit(
        self,
        data: dict,
        metadata: dict | None = None,
    ) -> tuple[Commitment, CommitmentOpening]:
        """
        create a commitment to data.

        returns both commitment (public) and opening (keep secret until reveal).
        """
        if not isinstance(data, dict):
            raise ValueError("data must be a dict")

        salt = self._generate_salt()
        canonical = self._canonical(data)
        digest = self._hash(canonical + salt)
        cid = self._generate_id()

        commitment = Commitment(
            commitment_id=cid,
            digest=digest,
            salt=salt,
            public_metadata=metadata or {},
            created_at=time.time(),
        )

        opening = CommitmentOpening(
            commitment_id=cid,
            data=data,
            salt=salt,
        )

        logger.debug(f"created commitment {cid}")
        return commitment, opening

    def verify(
        self,
        commitment: Commitment,
        opening: CommitmentOpening,
    ) -> VerificationResult:
        """
        verify that opening matches commitment.
        """
        if commitment.commitment_id != opening.commitment_id:
            return VerificationResult(
                valid=False,
                reason="commitment ID mismatch",
                commitment_id=commitment.commitment_id,
            )

        if commitment.salt != opening.salt:
            return VerificationResult(
                valid=False,
                reason="salt mismatch",
                commitment_id=commitment.commitment_id,
            )

        canonical = self._canonical(opening.data)
        expected_digest = self._hash(canonical + opening.salt)

        if expected_digest != commitment.digest:
            return VerificationResult(
                valid=False,
                reason="digest mismatch - data was modified",
                commitment_id=commitment.commitment_id,
            )

        return VerificationResult(
            valid=True,
            reason="commitment verified",
            commitment_id=commitment.commitment_id,
        )


class CommitmentVerifier:
    """
    standalone verifier - does not need access to original scheme.
    """

    def _canonical(self, data: dict) -> str:
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    def _hash(self, data: str) -> str:
        return hashlib.sha256(data.encode()).hexdigest()

    def verify(
        self,
        commitment: Commitment,
        opening: CommitmentOpening,
    ) -> VerificationResult:
        """verify commitment without needing the original scheme."""
        if commitment.commitment_id != opening.commitment_id:
            return VerificationResult(
                valid=False,
                reason="commitment ID mismatch",
                commitment_id=commitment.commitment_id,
            )

        canonical = self._canonical(opening.data)
        expected_digest = self._hash(canonical + opening.salt)

        if expected_digest != commitment.digest:
            return VerificationResult(
                valid=False,
                reason="digest mismatch",
                commitment_id=commitment.commitment_id,
            )

        return VerificationResult(
            valid=True,
            reason="verified",
            commitment_id=commitment.commitment_id,
        )


@dataclass
class StateTransitionCommitment:
    """
    commitment to a state transition.

    public: pre_state_digest, post_state_digest, transition_type
    committed: actual state contents (revealed on audit)
    """
    pre_state_digest: str
    post_state_digest: str
    input_digest: str
    commitment: Commitment
    transition_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pre_state_digest": self.pre_state_digest,
            "post_state_digest": self.post_state_digest,
            "input_digest": self.input_digest,
            "commitment": self.commitment.to_dict(),
            "transition_type": self.transition_type,
        }


class TransitionCommitmentScheme:
    """
    commitment scheme specialized for state transitions.
    """

    def __init__(self, seed: int = 42):
        self._scheme = CommitmentScheme(seed)

    def commit_transition(
        self,
        pre_state: dict,
        post_state: dict,
        input_data: dict,
        transition_type: str,
    ) -> tuple[StateTransitionCommitment, CommitmentOpening]:
        """commit to a state transition."""
        # compute public digests
        pre_digest = hashlib.sha256(
            json.dumps(pre_state, sort_keys=True).encode()
        ).hexdigest()
        post_digest = hashlib.sha256(
            json.dumps(post_state, sort_keys=True).encode()
        ).hexdigest()
        input_digest = hashlib.sha256(
            json.dumps(input_data, sort_keys=True).encode()
        ).hexdigest()

        # commit to full details
        full_data = {
            "pre_state": pre_state,
            "post_state": post_state,
            "input_data": input_data,
            "transition_type": transition_type,
        }
        metadata = {
            "pre_state_digest": pre_digest,
            "post_state_digest": post_digest,
            "input_digest": input_digest,
            "transition_type": transition_type,
        }

        commitment, opening = self._scheme.commit(full_data, metadata)

        return StateTransitionCommitment(
            pre_state_digest=pre_digest,
            post_state_digest=post_digest,
            input_digest=input_digest,
            commitment=commitment,
            transition_type=transition_type,
        ), opening

    def verify_transition(
        self,
        stc: StateTransitionCommitment,
        opening: CommitmentOpening,
    ) -> VerificationResult:
        """verify a transition commitment."""
        # first verify the base commitment
        result = CommitmentVerifier().verify(stc.commitment, opening)
        if not result.valid:
            return result

        # then verify metadata matches
        if opening.data.get("transition_type") != stc.transition_type:
            return VerificationResult(
                valid=False,
                reason="transition type mismatch",
                commitment_id=stc.commitment.commitment_id,
            )

        return VerificationResult(
            valid=True,
            reason="transition commitment verified",
            commitment_id=stc.commitment.commitment_id,
        )


class BatchCommitmentScheme:
    """
    batch multiple commitments into a single merkle root.
    """

    def __init__(self, seed: int = 42):
        self._scheme = CommitmentScheme(seed)

    def commit_batch(
        self,
        items: list[dict],
    ) -> tuple[list[Commitment], list[CommitmentOpening], str]:
        """
        commit to a batch of items.

        returns (commitments, openings, merkle_root)
        """
        from crypto.merkle import MerkleTree

        commitments = []
        openings = []

        for item in items:
            c, o = self._scheme.commit(item)
            commitments.append(c)
            openings.append(o)

        # build merkle tree of commitment digests
        tree = MerkleTree()
        for c in commitments:
            tree.add_leaf(c.digest)
        root = tree.build()

        return commitments, openings, root
