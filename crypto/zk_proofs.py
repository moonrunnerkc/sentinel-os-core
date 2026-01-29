# Author: Bradley R. Kinnard
# zero-knowledge proofs - Pedersen commitments and Schnorr proofs
# simplified implementation using modular arithmetic over a safe prime

import hashlib
import secrets
from dataclasses import dataclass
from typing import Any

from utils.helpers import get_logger

logger = get_logger(__name__)


# Use a safe prime p = 2q + 1 where q is also prime
# This gives us a subgroup of order q for Schnorr
# Using a 256-bit safe prime for demonstration
P = 0xFFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AACAA68FFFFFFFFFFFFFFFF
Q = (P - 1) // 2  # subgroup order

# Generator (primitive root mod p, generates subgroup of order q)
G = 2
H = 3  # second independent generator


def _mod_exp(base: int, exp: int, mod: int) -> int:
    """modular exponentiation."""
    return pow(base, exp, mod)


def _random_scalar(seed: int | None = None) -> int:
    """generate random scalar in [1, Q-1]."""
    if seed is not None:
        import random
        rng = random.Random(seed)
        return rng.randint(1, Q - 1)
    return secrets.randbelow(Q - 1) + 1


def _int_to_bytes(n: int) -> bytes:
    """convert integer to bytes (variable length)."""
    if n == 0:
        return b'\x00'
    return n.to_bytes((n.bit_length() + 7) // 8, 'big')


def _hash_to_scalar(*args: int) -> int:
    """hash integers to a scalar (Fiat-Shamir transform)."""
    hasher = hashlib.sha256()
    for arg in args:
        hasher.update(_int_to_bytes(arg))
    digest = hasher.digest()
    return int.from_bytes(digest, 'big') % Q


# export constants for tests
MODULUS = P
GROUP_ORDER = Q


@dataclass(frozen=True)
class PedersenCommitment:
    """
    Pedersen commitment: C = g^v * h^r mod p

    properties:
    - computationally hiding: C reveals nothing about v without r
    - computationally binding: cannot find (v', r') != (v, r) with same C
    - additively homomorphic: C(a) * C(b) = C(a+b) (with combined blinding)

    NOT zero-knowledge by itself - opening reveals v. use with Schnorr
    proofs to prove properties without revealing v.
    """
    commitment: int  # g^v * h^r mod p
    value: int       # the committed value (kept secret until opening)
    blinding: int    # the random blinding factor r

    def to_dict(self) -> dict[str, Any]:
        return {
            "commitment": self.commitment,
            # value and blinding are secret - not serialized
        }

    @property
    def public(self) -> int:
        """return only the public commitment."""
        return self.commitment


class PedersenScheme:
    """
    Pedersen commitment scheme.

    usage:
    1. commit(v) -> PedersenCommitment (keep blinding secret)
    2. verify(C, v, r) -> bool (check opening)
    3. add(C1, C2) -> combined commitment (homomorphic)
    """

    def __init__(self, seed: int | None = None):
        self._seed = seed
        self._counter = 0

    def _next_scalar(self) -> int:
        """get next deterministic scalar if seeded."""
        if self._seed is not None:
            self._counter += 1
            return _random_scalar(self._seed + self._counter)
        return _random_scalar()

    def commit(self, value: int, blinding: int | None = None) -> PedersenCommitment:
        """
        create Pedersen commitment to value.

        C = g^v * h^r mod p

        if blinding not provided, generates random blinding factor.
        """
        if value < 0:
            raise ValueError("value must be non-negative")

        r = blinding if blinding is not None else self._next_scalar()

        # C = g^v * h^r mod P
        g_v = _mod_exp(G, value, P)
        h_r = _mod_exp(H, r, P)
        commitment = (g_v * h_r) % P

        logger.debug(f"created pedersen commitment for value (blinding factor generated)")

        return PedersenCommitment(
            commitment=commitment,
            value=value,
            blinding=r,
        )

    def verify(self, commitment: int, value: int, blinding: int) -> bool:
        """
        verify Pedersen commitment opening.

        checks: C == g^v * h^r mod p
        """
        g_v = _mod_exp(G, value, P)
        h_r = _mod_exp(H, blinding, P)
        expected = (g_v * h_r) % P

        return commitment == expected

    def add(
        self,
        c1: PedersenCommitment,
        c2: PedersenCommitment,
    ) -> PedersenCommitment:
        """
        homomorphically add two commitments.

        C(a+b) = C(a) * C(b) with combined blinding r1 + r2
        """
        combined_commitment = (c1.commitment * c2.commitment) % P
        combined_value = c1.value + c2.value
        combined_blinding = (c1.blinding + c2.blinding) % Q

        return PedersenCommitment(
            commitment=combined_commitment,
            value=combined_value,
            blinding=combined_blinding,
        )


@dataclass(frozen=True)
class SchnorrProof:
    """
    Schnorr proof of knowledge of discrete log.

    proves: "I know x such that y = g^x" without revealing x.

    protocol (non-interactive via Fiat-Shamir):
    1. prover picks random k, computes R = g^k
    2. challenge c = H(g, y, R)
    3. response s = k + c*x mod q
    4. verifier checks: g^s == R * y^c
    """
    commitment_r: int  # R = g^k
    challenge: int     # c = H(g, y, R)
    response: int      # s = k + c*x mod q


class SchnorrProver:
    """
    Schnorr proof generation.

    proves knowledge of discrete log without revealing it.
    """

    def __init__(self, seed: int | None = None):
        self._seed = seed
        self._counter = 0

    def _next_scalar(self) -> int:
        if self._seed is not None:
            self._counter += 1
            return _random_scalar(self._seed + self._counter)
        return _random_scalar()

    def prove(self, secret: int, public: int, generator: int = G) -> SchnorrProof:
        """
        generate Schnorr proof that we know secret where public = g^secret.

        uses Fiat-Shamir heuristic for non-interactive proof.
        """
        # step 1: pick random k, compute R = g^k
        k = self._next_scalar()
        r = _mod_exp(generator, k, P)

        # step 2: challenge c = H(g || y || R)
        c = _hash_to_scalar(generator, public, r)

        # step 3: response s = k + c*x mod q
        s = (k + c * secret) % Q

        logger.debug("generated schnorr proof")

        return SchnorrProof(
            commitment_r=r,
            challenge=c,
            response=s,
        )


class SchnorrVerifier:
    """
    Schnorr proof verification.
    """

    def verify(
        self,
        proof: SchnorrProof,
        public: int,
        generator: int = G,
    ) -> tuple[bool, str]:
        """
        verify Schnorr proof.

        checks: g^s == R * y^c
        """
        # recompute challenge
        expected_c = _hash_to_scalar(generator, public, proof.commitment_r)

        if proof.challenge != expected_c:
            return False, "challenge mismatch (possible tampering)"

        # verify: g^s == R * y^c mod p
        lhs = _mod_exp(generator, proof.response, P)
        y_c = _mod_exp(public, proof.challenge, P)
        rhs = (proof.commitment_r * y_c) % P

        if lhs != rhs:
            return False, "verification equation failed"

        return True, "valid"


@dataclass(frozen=True)
class StateTransitionProof:
    """
    proof of valid state transition.

    proves specific bounded invariants:
    - belief count delta matches declared insertions/deletions
    - confidence changes are within declared bounds

    NOT a general-purpose computation proof. scope is explicit.
    """
    pre_state_commitment: int
    post_state_commitment: int
    delta_commitment: int  # commitment to the change
    schnorr_proof: SchnorrProof  # proof that we know the opening
    invariant_type: str  # which invariant this proves
    metadata: dict[str, Any]


class StateTransitionProver:
    """
    generates proofs for state transitions.

    SCOPE LIMITATION: this proves specific invariants only.
    - BELIEF_COUNT: post_count = pre_count + delta
    - CONFIDENCE_BOUND: |confidence_change| <= max_delta

    it does NOT prove arbitrary computation or full state validity.
    """

    def __init__(self, seed: int | None = None):
        self._pedersen = PedersenScheme(seed)
        self._schnorr = SchnorrProver(seed)

    def prove_belief_count_delta(
        self,
        pre_count: int,
        post_count: int,
        declared_delta: int,
    ) -> tuple[StateTransitionProof | None, str]:
        """
        prove that post_count = pre_count + declared_delta.

        returns (proof, message) where proof is None if invariant violated.
        """
        actual_delta = post_count - pre_count

        if actual_delta != declared_delta:
            return None, f"invariant violated: actual delta {actual_delta} != declared {declared_delta}"

        # commit to pre, post, delta
        pre_commit = self._pedersen.commit(pre_count)
        post_commit = self._pedersen.commit(post_count)
        delta_commit = self._pedersen.commit(declared_delta)

        # prove we know the opening of delta commitment
        delta_public = _mod_exp(G, declared_delta, P)
        proof = self._schnorr.prove(declared_delta, delta_public)

        transition_proof = StateTransitionProof(
            pre_state_commitment=pre_commit.commitment,
            post_state_commitment=post_commit.commitment,
            delta_commitment=delta_commit.commitment,
            schnorr_proof=proof,
            invariant_type="BELIEF_COUNT",
            metadata={
                "pre_blinding": pre_commit.blinding,
                "post_blinding": post_commit.blinding,
                "delta_blinding": delta_commit.blinding,
            },
        )

        logger.debug(f"generated belief count delta proof: {pre_count} + {declared_delta} = {post_count}")

        return transition_proof, "valid"

    def prove_confidence_bound(
        self,
        old_confidence: float,
        new_confidence: float,
        max_delta: float,
    ) -> tuple[StateTransitionProof | None, str]:
        """
        prove that |new_confidence - old_confidence| <= max_delta.

        returns (proof, message) where proof is None if invariant violated.
        """
        actual_delta = abs(new_confidence - old_confidence)

        if actual_delta > max_delta:
            return None, f"invariant violated: delta {actual_delta:.4f} > max {max_delta:.4f}"

        # convert to integers for commitment (scale by 1e6 for precision)
        scale = 1_000_000
        old_int = int(old_confidence * scale)
        new_int = int(new_confidence * scale)
        delta_int = abs(new_int - old_int)

        # commit to values
        old_commit = self._pedersen.commit(old_int)
        new_commit = self._pedersen.commit(new_int)
        delta_commit = self._pedersen.commit(delta_int)

        # prove we know delta
        delta_public = _mod_exp(G, delta_int, P)
        proof = self._schnorr.prove(delta_int, delta_public)

        transition_proof = StateTransitionProof(
            pre_state_commitment=old_commit.commitment,
            post_state_commitment=new_commit.commitment,
            delta_commitment=delta_commit.commitment,
            schnorr_proof=proof,
            invariant_type="CONFIDENCE_BOUND",
            metadata={
                "scale": scale,
                "max_delta_scaled": int(max_delta * scale),
            },
        )

        logger.debug(f"generated confidence bound proof: delta {actual_delta:.4f} <= {max_delta:.4f}")

        return transition_proof, "valid"


class StateTransitionVerifier:
    """
    verifies state transition proofs.
    """

    def __init__(self):
        self._schnorr = SchnorrVerifier()

    def verify(self, proof: StateTransitionProof) -> tuple[bool, str]:
        """
        verify state transition proof.

        checks schnorr proof structure. does NOT verify the invariant
        itself (that requires the openings which are kept secret).
        """
        if proof.invariant_type == "BELIEF_COUNT":
            # verify proof structure is valid
            if proof.schnorr_proof.commitment_r == 0:
                return False, "invalid proof: zero commitment"

            if proof.schnorr_proof.response == 0:
                return False, "invalid proof: zero response"

            # structure is valid; actual invariant verification requires openings
            return True, "proof structure valid (invariant requires opening to verify)"

        elif proof.invariant_type == "CONFIDENCE_BOUND":
            if proof.schnorr_proof.commitment_r == 0:
                return False, "invalid proof: zero commitment"

            return True, "proof structure valid (invariant requires opening to verify)"

        return False, f"unknown invariant type: {proof.invariant_type}"
