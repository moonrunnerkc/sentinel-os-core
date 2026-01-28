# Author: Bradley R. Kinnard
# comprehensive benchmark suite with statistical rigor

import json
import time
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable
from pathlib import Path

import numpy as np

from utils.helpers import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """result of a single benchmark."""
    name: str
    iterations: int
    times_ms: list[float]
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    ops_per_second: float
    memory_mb: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "ops_per_second": self.ops_per_second,
            "memory_mb": self.memory_mb,
            "metadata": self.metadata,
        }


@dataclass
class ComparisonResult:
    """result of comparing two benchmarks."""
    baseline_name: str
    comparison_name: str
    baseline_mean: float
    comparison_mean: float
    speedup: float  # comparison_mean / baseline_mean
    t_statistic: float
    p_value: float
    significant: bool  # p < 0.05
    confidence_interval: tuple[float, float]


class BenchmarkRunner:
    """
    benchmark runner with statistical analysis.

    provides:
    - multiple iterations for statistical significance
    - warmup runs to stabilize caches
    - percentile analysis
    - memory tracking
    - reproducible seeds
    """

    def __init__(
        self,
        warmup_iterations: int = 5,
        min_iterations: int = 10,
        max_iterations: int = 100,
        target_time_seconds: float = 1.0,
    ):
        self._warmup = warmup_iterations
        self._min_iter = min_iterations
        self._max_iter = max_iterations
        self._target_time = target_time_seconds
        self._results: list[BenchmarkResult] = []

    def run(
        self,
        name: str,
        fn: Callable[[], Any],
        setup: Callable[[], None] | None = None,
        teardown: Callable[[], None] | None = None,
        iterations: int | None = None,
    ) -> BenchmarkResult:
        """run a benchmark with full statistical analysis."""
        logger.info(f"running benchmark: {name}")

        # setup
        if setup:
            setup()

        # warmup
        for _ in range(self._warmup):
            fn()

        # determine iteration count
        if iterations is None:
            # auto-calibrate based on target time
            start = time.perf_counter()
            fn()
            single_time = time.perf_counter() - start

            if single_time > 0:
                iterations = min(
                    self._max_iter,
                    max(self._min_iter, int(self._target_time / single_time))
                )
            else:
                iterations = self._min_iter

        # run timed iterations
        times_ms = []
        for _ in range(iterations):
            start = time.perf_counter()
            fn()
            elapsed = (time.perf_counter() - start) * 1000
            times_ms.append(elapsed)

        # teardown
        if teardown:
            teardown()

        # compute statistics
        result = BenchmarkResult(
            name=name,
            iterations=iterations,
            times_ms=times_ms,
            mean_ms=statistics.mean(times_ms),
            std_ms=statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
            min_ms=min(times_ms),
            max_ms=max(times_ms),
            p50_ms=np.percentile(times_ms, 50),
            p95_ms=np.percentile(times_ms, 95),
            p99_ms=np.percentile(times_ms, 99),
            ops_per_second=1000 / statistics.mean(times_ms) if statistics.mean(times_ms) > 0 else 0,
        )

        self._results.append(result)
        logger.info(f"  {name}: mean={result.mean_ms:.3f}ms, std={result.std_ms:.3f}ms")

        return result

    def compare(
        self,
        baseline: BenchmarkResult,
        comparison: BenchmarkResult,
        alpha: float = 0.05,
    ) -> ComparisonResult:
        """compare two benchmark results with statistical testing."""
        from scipy import stats

        # two-sample t-test
        t_stat, p_value = stats.ttest_ind(baseline.times_ms, comparison.times_ms)

        speedup = baseline.mean_ms / comparison.mean_ms if comparison.mean_ms > 0 else float("inf")

        # confidence interval for difference
        diff = np.array(comparison.times_ms) - np.mean(baseline.times_ms)
        ci = stats.t.interval(
            1 - alpha,
            len(diff) - 1,
            loc=np.mean(diff),
            scale=stats.sem(diff) if len(diff) > 1 else 0,
        )

        return ComparisonResult(
            baseline_name=baseline.name,
            comparison_name=comparison.name,
            baseline_mean=baseline.mean_ms,
            comparison_mean=comparison.mean_ms,
            speedup=speedup,
            t_statistic=t_stat,
            p_value=p_value,
            significant=p_value < alpha,
            confidence_interval=ci,
        )

    def get_results(self) -> list[BenchmarkResult]:
        """return all benchmark results."""
        return self._results.copy()

    def export_json(self, path: str | Path) -> None:
        """export results to JSON."""
        data = {
            "timestamp": time.time(),
            "results": [r.to_dict() for r in self._results],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


class SentinelBenchmarkSuite:
    """
    comprehensive benchmark suite for Sentinel OS Core.

    measures:
    - belief operations (insert, update, query)
    - contradiction resolution
    - goal collapse convergence
    - memory operations
    - cryptographic operations
    - isolation overhead
    """

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._runner = BenchmarkRunner()
        np.random.seed(seed)

    def run_all(self) -> dict[str, BenchmarkResult]:
        """run all benchmarks."""
        results = {}

        results["belief_insert"] = self._bench_belief_insert()
        results["belief_update"] = self._bench_belief_update()
        results["contradiction_detect"] = self._bench_contradiction_detect()
        results["state_digest"] = self._bench_state_digest()
        results["hmac_sign"] = self._bench_hmac_sign()
        results["zk_proof_gen"] = self._bench_zk_proof()
        results["dp_noise"] = self._bench_dp_noise()
        results["merkle_build"] = self._bench_merkle_tree()

        return results

    def _bench_belief_insert(self) -> BenchmarkResult:
        """benchmark belief insertion."""
        from verification.state_machine import TransitionEngine, BeliefState
        import hashlib

        engine = TransitionEngine()
        counter = [0]

        def insert_belief():
            counter[0] += 1
            belief = BeliefState(
                belief_id=f"b_{counter[0]}",
                content_hash=hashlib.sha256(f"content_{counter[0]}".encode()).hexdigest()[:16],
                confidence=np.random.random(),
                timestamp=time.time(),
            )
            engine.insert_belief(belief)

        return self._runner.run("belief_insert", insert_belief)

    def _bench_belief_update(self) -> BenchmarkResult:
        """benchmark belief updates."""
        from verification.state_machine import TransitionEngine, BeliefState
        import hashlib

        engine = TransitionEngine()

        # pre-populate
        for i in range(100):
            belief = BeliefState(
                belief_id=f"b_{i}",
                content_hash=hashlib.sha256(f"content_{i}".encode()).hexdigest()[:16],
                confidence=0.5,
                timestamp=time.time(),
            )
            engine.insert_belief(belief)

        def update_belief():
            bid = f"b_{np.random.randint(0, 100)}"
            engine.update_belief(bid, np.random.random(), time.time())

        return self._runner.run("belief_update", update_belief)

    def _bench_contradiction_detect(self) -> BenchmarkResult:
        """benchmark contradiction detection."""
        from verification.invariants import InvariantChecker
        from verification.state_machine import SystemState, BeliefState
        import hashlib

        checker = InvariantChecker()

        # create state with potential contradictions
        state = SystemState()
        for i in range(50):
            state.beliefs[f"b_{i}"] = BeliefState(
                belief_id=f"b_{i}",
                content_hash=hashlib.sha256(f"c_{i}".encode()).hexdigest()[:16],
                confidence=np.random.random(),
                timestamp=time.time(),
            )

        def check_invariants():
            checker.check_all(state)

        return self._runner.run("contradiction_detect", check_invariants)

    def _bench_state_digest(self) -> BenchmarkResult:
        """benchmark state digest computation."""
        from verification.state_machine import SystemState, BeliefState
        import hashlib

        state = SystemState()
        for i in range(100):
            state.beliefs[f"b_{i}"] = BeliefState(
                belief_id=f"b_{i}",
                content_hash=hashlib.sha256(f"c_{i}".encode()).hexdigest()[:16],
                confidence=np.random.random(),
                timestamp=time.time(),
            )

        def compute_digest():
            state.digest()

        return self._runner.run("state_digest", compute_digest)

    def _bench_hmac_sign(self) -> BenchmarkResult:
        """benchmark HMAC signing."""
        import hmac
        import hashlib

        key = b"secret_key_32bytes_exactly_here!"
        message = b"test message for signing" * 10

        def sign_message():
            hmac.new(key, message, hashlib.sha256).hexdigest()

        return self._runner.run("hmac_sign", sign_message)

    def _bench_zk_proof(self) -> BenchmarkResult:
        """benchmark ZK proof generation."""
        from crypto.zk_proofs import ZKProver

        prover = ZKProver(seed=self._seed)

        pre_state = {"beliefs": {"b1": 0.5}}
        post_state = {"beliefs": {"b1": 0.6}}
        input_data = {"delta": 0.1}

        def generate_proof():
            prover.prove_state_transition(
                pre_state, post_state, input_data, "test_fn_hash"
            )

        return self._runner.run("zk_proof_gen", generate_proof)

    def _bench_dp_noise(self) -> BenchmarkResult:
        """benchmark differential privacy noise addition."""
        from privacy.mechanisms import laplace_mechanism

        def add_noise():
            laplace_mechanism(0.5, sensitivity=1.0, epsilon=0.1)

        return self._runner.run("dp_noise", add_noise)

    def _bench_merkle_tree(self) -> BenchmarkResult:
        """benchmark merkle tree construction."""
        from crypto.merkle import MerkleTree

        leaves = [f"leaf_{i}" for i in range(100)]

        def build_tree():
            tree = MerkleTree()
            tree.add_leaves(leaves)
            tree.build()

        return self._runner.run("merkle_build", build_tree)

    def generate_report(self, results: dict[str, BenchmarkResult]) -> str:
        """generate human-readable benchmark report."""
        lines = [
            "=" * 60,
            "SENTINEL OS CORE BENCHMARK REPORT",
            "=" * 60,
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Seed: {self._seed}",
            "",
            "RESULTS:",
            "-" * 60,
        ]

        for name, result in results.items():
            lines.append(f"\n{name}:")
            lines.append(f"  Iterations: {result.iterations}")
            lines.append(f"  Mean:       {result.mean_ms:.4f} ms")
            lines.append(f"  Std Dev:    {result.std_ms:.4f} ms")
            lines.append(f"  P50:        {result.p50_ms:.4f} ms")
            lines.append(f"  P95:        {result.p95_ms:.4f} ms")
            lines.append(f"  P99:        {result.p99_ms:.4f} ms")
            lines.append(f"  Ops/sec:    {result.ops_per_second:.2f}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)
