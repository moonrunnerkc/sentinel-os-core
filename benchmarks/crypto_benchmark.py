# Author: Bradley R. Kinnard
# crypto benchmark suite - deterministic benchmarks for ZK, PQ, HE operations

import time
import statistics
from dataclasses import dataclass
from typing import Any, Callable

from utils.helpers import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """result of a single benchmark."""
    name: str
    iterations: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    ops_per_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "mean_ms": round(self.mean_ms, 4),
            "std_ms": round(self.std_ms, 4),
            "min_ms": round(self.min_ms, 4),
            "max_ms": round(self.max_ms, 4),
            "p50_ms": round(self.p50_ms, 4),
            "p95_ms": round(self.p95_ms, 4),
            "p99_ms": round(self.p99_ms, 4),
            "ops_per_sec": round(self.ops_per_sec, 2),
        }


def run_benchmark(
    name: str,
    fn: Callable[[], Any],
    iterations: int = 100,
    warmup: int = 5,
) -> BenchmarkResult:
    """run a benchmark and collect statistics."""
    # warmup
    for _ in range(warmup):
        fn()

    # timed runs
    times_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed)

    times_ms.sort()
    mean = statistics.mean(times_ms)
    std = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0

    p50_idx = int(0.50 * len(times_ms))
    p95_idx = int(0.95 * len(times_ms))
    p99_idx = int(0.99 * len(times_ms))

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        mean_ms=mean,
        std_ms=std,
        min_ms=times_ms[0],
        max_ms=times_ms[-1],
        p50_ms=times_ms[p50_idx],
        p95_ms=times_ms[min(p95_idx, len(times_ms) - 1)],
        p99_ms=times_ms[min(p99_idx, len(times_ms) - 1)],
        ops_per_sec=1000.0 / mean if mean > 0 else 0.0,
    )


def bench_pedersen_commit(iterations: int = 100) -> BenchmarkResult:
    """benchmark Pedersen commitment generation."""
    from crypto.zk_proofs import PedersenScheme

    scheme = PedersenScheme(seed=42)
    value = 1000

    def fn():
        scheme.commit(value)

    return run_benchmark("pedersen_commit", fn, iterations)


def bench_pedersen_verify(iterations: int = 100) -> BenchmarkResult:
    """benchmark Pedersen commitment verification."""
    from crypto.zk_proofs import PedersenScheme

    scheme = PedersenScheme(seed=42)
    commitment = scheme.commit(1000)

    def fn():
        scheme.verify(commitment.commitment, commitment.value, commitment.blinding)

    return run_benchmark("pedersen_verify", fn, iterations)


def bench_schnorr_prove(iterations: int = 100) -> BenchmarkResult:
    """benchmark Schnorr proof generation."""
    from crypto.zk_proofs import SchnorrProver, G, P, _mod_exp

    prover = SchnorrProver(seed=42)
    secret = 12345
    public = _mod_exp(G, secret, P)

    def fn():
        prover.prove(secret, public)

    return run_benchmark("schnorr_prove", fn, iterations)


def bench_schnorr_verify(iterations: int = 100) -> BenchmarkResult:
    """benchmark Schnorr proof verification."""
    from crypto.zk_proofs import SchnorrProver, SchnorrVerifier, G, P, _mod_exp

    prover = SchnorrProver(seed=42)
    verifier = SchnorrVerifier()
    secret = 12345
    public = _mod_exp(G, secret, P)
    proof = prover.prove(secret, public)

    def fn():
        verifier.verify(proof, public)

    return run_benchmark("schnorr_verify", fn, iterations)


def bench_state_transition_proof(iterations: int = 100) -> BenchmarkResult:
    """benchmark state transition proof generation."""
    from crypto.zk_proofs import StateTransitionProver

    prover = StateTransitionProver(seed=42)

    def fn():
        prover.prove_belief_count_delta(10, 15, 5)

    return run_benchmark("state_transition_proof", fn, iterations)


def bench_ed25519_sign(iterations: int = 100) -> BenchmarkResult:
    """benchmark Ed25519 signing."""
    from crypto.pq_signatures import generate_keypair, Signer, Algorithm

    keypair = generate_keypair(Algorithm.ED25519)
    signer = Signer(keypair)
    message = b"benchmark message for signing"

    def fn():
        signer.sign(message)

    return run_benchmark("ed25519_sign", fn, iterations)


def bench_ed25519_verify(iterations: int = 100) -> BenchmarkResult:
    """benchmark Ed25519 verification."""
    from crypto.pq_signatures import generate_keypair, Signer, Verifier, Algorithm

    keypair = generate_keypair(Algorithm.ED25519)
    signer = Signer(keypair)
    verifier = Verifier.from_keypair(keypair)
    message = b"benchmark message for signing"
    signature = signer.sign(message)

    def fn():
        verifier.verify(message, signature)

    return run_benchmark("ed25519_verify", fn, iterations)


def bench_merkle_build(iterations: int = 100, n_leaves: int = 100) -> BenchmarkResult:
    """benchmark merkle tree construction."""
    from crypto.merkle import MerkleTree

    leaves = [f"leaf_{i}" for i in range(n_leaves)]

    def fn():
        tree = MerkleTree()
        tree.add_leaves(leaves)
        tree.build()

    return run_benchmark(f"merkle_build_{n_leaves}", fn, iterations)


def bench_merkle_proof(iterations: int = 100) -> BenchmarkResult:
    """benchmark merkle proof generation."""
    from crypto.merkle import MerkleTree

    tree = MerkleTree()
    tree.add_leaves([f"leaf_{i}" for i in range(100)])
    tree.build()

    def fn():
        tree.get_proof(50)

    return run_benchmark("merkle_proof", fn, iterations)


def bench_he_encrypt(iterations: int = 50) -> BenchmarkResult | None:
    """benchmark HE encryption (requires tenseal)."""
    from crypto.homomorphic import tenseal_available

    if not tenseal_available():
        logger.warning("skipping HE benchmark - tenseal not available")
        return None

    from crypto.homomorphic import HomomorphicEngine

    engine = HomomorphicEngine()
    value = 0.5

    def fn():
        engine.encrypt_scalar(value)

    return run_benchmark("he_encrypt", fn, iterations)


def bench_he_add(iterations: int = 50) -> BenchmarkResult | None:
    """benchmark HE addition (requires tenseal)."""
    from crypto.homomorphic import tenseal_available

    if not tenseal_available():
        return None

    from crypto.homomorphic import HomomorphicEngine

    engine = HomomorphicEngine()
    a = engine.encrypt_scalar(0.5)
    b = engine.encrypt_scalar(0.3)

    def fn():
        engine.add(a, b)

    return run_benchmark("he_add", fn, iterations)


def bench_he_decrypt(iterations: int = 50) -> BenchmarkResult | None:
    """benchmark HE decryption (requires tenseal)."""
    from crypto.homomorphic import tenseal_available

    if not tenseal_available():
        return None

    from crypto.homomorphic import HomomorphicEngine

    engine = HomomorphicEngine()
    encrypted = engine.encrypt_scalar(0.5)

    def fn():
        engine.decrypt_scalar(encrypted)

    return run_benchmark("he_decrypt", fn, iterations)


def run_all_benchmarks(iterations: int = 100) -> list[BenchmarkResult]:
    """run all crypto benchmarks."""
    results = []

    # ZK benchmarks
    results.append(bench_pedersen_commit(iterations))
    results.append(bench_pedersen_verify(iterations))
    results.append(bench_schnorr_prove(iterations))
    results.append(bench_schnorr_verify(iterations))
    results.append(bench_state_transition_proof(iterations))

    # signature benchmarks
    results.append(bench_ed25519_sign(iterations))
    results.append(bench_ed25519_verify(iterations))

    # merkle benchmarks
    results.append(bench_merkle_build(iterations, n_leaves=100))
    results.append(bench_merkle_proof(iterations))

    # HE benchmarks (if available)
    he_encrypt = bench_he_encrypt(iterations // 2)
    if he_encrypt:
        results.append(he_encrypt)

    he_add = bench_he_add(iterations // 2)
    if he_add:
        results.append(he_add)

    he_decrypt = bench_he_decrypt(iterations // 2)
    if he_decrypt:
        results.append(he_decrypt)

    return results


def print_benchmark_results(results: list[BenchmarkResult]) -> None:
    """print benchmark results in table format."""
    print("\n" + "=" * 80)
    print("CRYPTO BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Operation':<30} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Ops/sec':<12}")
    print("-" * 80)

    for r in results:
        print(f"{r.name:<30} {r.mean_ms:<12.4f} {r.p95_ms:<12.4f} {r.ops_per_sec:<12.2f}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    results = run_all_benchmarks(iterations=100)
    print_benchmark_results(results)
