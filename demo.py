#!/usr/bin/env python3
# Author: Bradley R. Kinnard
# sentinel os core demo - run with: python demo.py

import time

def main():
    print("=" * 60)
    print("SENTINEL OS CORE - DEMO")
    print("=" * 60)
    print()

    # 1. verification
    print("[1/5] Verification Layer...")
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
    print("[2/5] Privacy Layer...")
    from privacy.budget import PrivacyAccountant
    from privacy.mechanisms import laplace_mechanism

    accountant = PrivacyAccountant(total_epsilon=1.0, total_delta=1e-5)
    accountant.spend(0.1, mechanism="laplace", operation="belief_update")
    noisy_value, _ = laplace_mechanism(0.5, sensitivity=1.0, epsilon=0.1)
    print(f"      Budget remaining: {accountant.budget.remaining_epsilon():.2f} epsilon")
    print(f"      Noisy value: {noisy_value:.4f} (original: 0.5)")
    print()

    # 3. zk proofs
    print("[3/5] ZK Proofs...")
    from crypto.zk_proofs import ZKProver, ZKVerifier

    prover = ZKProver(seed=42)
    verifier = ZKVerifier()
    proof = prover.prove_state_transition(
        pre_state={"belief": 0.5},
        post_state={"belief": 0.6},
        input_data={"delta": 0.1},
        transition_fn_hash="demo_fn",
    )
    valid, _ = verifier.verify(proof)
    print(f"      Proof valid: {valid}")
    print()

    # 4. merkle tree
    print("[4/5] Merkle Tree...")
    from crypto.merkle import MerkleTree

    tree = MerkleTree()
    tree.add_leaves(["belief_1", "belief_2", "belief_3", "goal_1"])
    root = tree.build()
    print(f"      Root: {root[:32]}...")
    print(f"      Leaves: 4")
    print()

    # 5. signed log chain
    print("[5/5] Signed Audit Chain...")
    from crypto.pq_signatures import generate_keypair, PQSigner, PQVerifier, SignedLogChain

    keypair = generate_keypair("demo_key")
    signer = PQSigner(keypair)
    chain = SignedLogChain(signer)

    chain.append({"event": "system_start", "timestamp": time.time()})
    chain.append({"event": "belief_insert", "id": "b1"})
    chain.append({"event": "goal_update", "id": "g1"})

    valid, _ = chain.verify_chain(PQVerifier.from_keypair(keypair))
    print(f"      Chain valid: {valid}")
    print(f"      Entries: {len(chain.get_chain())}")
    print()

    print("=" * 60)
    print("ALL SYSTEMS OPERATIONAL")
    print("=" * 60)


if __name__ == "__main__":
    main()
