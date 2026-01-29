#!/usr/bin/env python3
# Author: Bradley R. Kinnard
# sentinel os core demo - run with: python demo.py

import time
import json
from pathlib import Path


def load_config():
    """load system config and security rules."""
    import yaml

    config_path = Path("config/system_config.yaml")
    security_path = Path("config/security_rules.json")

    with open(config_path) as f:
        system_config = yaml.safe_load(f)

    with open(security_path) as f:
        security_rules = json.load(f)

    return system_config, security_rules


def print_crypto_mode(system_config: dict, security_rules: dict):
    """print current crypto configuration."""
    from crypto.pq_signatures import liboqs_available
    from crypto.homomorphic import tenseal_available

    sig_algo = security_rules.get("signature_algorithm", "ed25519")
    zk_enabled = system_config.get("features", {}).get("zk_proofs", True)
    he_enabled = system_config.get("features", {}).get("homomorphic_encryption", False)

    pq_status = "available" if liboqs_available() else "NOT installed"
    he_status = "available" if tenseal_available() else "NOT installed"

    print("[CRYPTO MODE]")
    print(f"  Signature algorithm: {sig_algo}")
    print(f"  ZK proofs (Pedersen/Schnorr): {'enabled' if zk_enabled else 'disabled'}")
    print(f"  Homomorphic encryption: {'enabled' if he_enabled else 'disabled'}")
    print(f"  liboqs-python: {pq_status}")
    print(f"  tenseal: {he_status}")
    print()


def main():
    print("=" * 60)
    print("SENTINEL OS CORE - DEMO")
    print("=" * 60)
    print()

    # load config
    system_config, security_rules = load_config()
    print_crypto_mode(system_config, security_rules)

    # 1. verification
    print("[1/6] Verification Layer...")
    from verification.state_machine import TransitionEngine, BeliefState

    engine = TransitionEngine()
    belief = BeliefState(
        belief_id="b1",
        content_hash="abc123",
        confidence=0.8,
        timestamp=time.time(),
    )
    engine.insert_belief(belief)
    valid, msg = engine.verify_trace_integrity()
    print(f"      Trace integrity: {valid}")
    print()

    # 2. privacy
    print("[2/6] Privacy Layer...")
    from privacy.budget import PrivacyAccountant
    from privacy.mechanisms import laplace_mechanism

    accountant = PrivacyAccountant(total_epsilon=1.0, total_delta=1e-5)
    accountant.spend(0.1, mechanism="laplace", operation="belief_update")
    noisy_value, _ = laplace_mechanism(0.5, sensitivity=1.0, epsilon=0.1)
    print(f"      Budget remaining: {accountant.budget.remaining_epsilon():.2f} epsilon")
    print(f"      Noisy value: {noisy_value:.4f} (original: 0.5)")
    print()

    # 3. state commitments
    print("[3/6] State Commitments...")
    from crypto.commitments import TransitionCommitmentScheme

    scheme = TransitionCommitmentScheme(seed=42)
    stc, opening = scheme.commit_transition(
        pre_state={"belief": 0.5},
        post_state={"belief": 0.6},
        input_data={"delta": 0.1},
        transition_type="demo_update",
    )
    result = scheme.verify_transition(stc, opening)
    print(f"      Commitment valid: {result.valid}")
    print(f"      Transition type: {stc.transition_type}")
    print()

    # 4. ZK proofs (Pedersen + Schnorr)
    print("[4/6] Zero-Knowledge Proofs...")
    from crypto.zk_proofs import (
        PedersenScheme,
        SchnorrProver,
        SchnorrVerifier,
        StateTransitionProver,
        StateTransitionVerifier,
        G,
        P,
        _mod_exp,
    )

    # Pedersen commitment
    pedersen = PedersenScheme(seed=42)
    commitment = pedersen.commit(100)
    pedersen_valid = pedersen.verify(
        commitment.commitment,
        commitment.value,
        commitment.blinding,
    )
    print(f"      Pedersen commitment valid: {pedersen_valid}")

    # Schnorr proof
    secret = 12345
    public = _mod_exp(G, secret, P)
    prover = SchnorrProver(seed=42)
    verifier = SchnorrVerifier()
    proof = prover.prove(secret, public)
    schnorr_valid, _ = verifier.verify(proof, public)
    print(f"      Schnorr proof valid: {schnorr_valid}")

    # State transition proof
    transition_prover = StateTransitionProver(seed=42)
    transition_proof, msg = transition_prover.prove_belief_count_delta(10, 15, 5)
    transition_verifier = StateTransitionVerifier()
    trans_valid, _ = transition_verifier.verify(transition_proof)
    print(f"      Belief count delta proof valid: {trans_valid}")
    print(f"      [SCOPE: Pedersen commitments + Schnorr proofs over discrete log]")
    print()

    # 5. merkle tree
    print("[5/6] Merkle Tree...")
    from crypto.merkle import MerkleTree

    tree = MerkleTree()
    tree.add_leaves(["belief_1", "belief_2", "belief_3", "goal_1"])
    root = tree.build()
    print(f"      Root: {root[:32]}...")
    print(f"      Leaves: 4")
    print()

    # 6. signed log chain
    print("[6/6] Signed Audit Chain...")
    from crypto.pq_signatures import generate_keypair, Signer, Verifier, SignedLogChain, Algorithm

    sig_algo = security_rules.get("signature_algorithm", "ed25519")
    algo_enum = Algorithm.ED25519  # default, safe fallback for demo

    keypair = generate_keypair(algo_enum, key_id="demo_key")
    signer = Signer(keypair)
    chain = SignedLogChain(signer)

    chain.append({"event": "system_start", "timestamp": time.time()})
    chain.append({"event": "belief_insert", "id": "b1"})
    chain.append({"event": "goal_update", "id": "g1"})

    valid, _ = chain.verify_chain(Verifier.from_keypair(keypair))
    print(f"      Chain valid: {valid}")
    print(f"      Algorithm: {keypair.algorithm.value}")
    print(f"      Entries: {len(chain.get_chain())}")
    print()

    print("=" * 60)
    print("ALL SYSTEMS OPERATIONAL")
    print("=" * 60)


if __name__ == "__main__":
    main()
