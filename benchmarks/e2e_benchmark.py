# Author: Bradley R. Kinnard
# end-to-end benchmarks - realistic workloads, reproducible offline

import time
import tempfile
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from utils.helpers import get_logger

logger = get_logger(__name__)


@dataclass
class E2EBenchmarkResult:
    """result of an end-to-end benchmark."""
    name: str
    description: str
    duration_ms: float
    operations: int
    ops_per_second: float
    memory_delta_bytes: int
    storage_bytes: int
    seed: int
    passed: bool
    threshold_ms: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "duration_ms": self.duration_ms,
            "operations": self.operations,
            "ops_per_second": self.ops_per_second,
            "memory_delta_bytes": self.memory_delta_bytes,
            "storage_bytes": self.storage_bytes,
            "seed": self.seed,
            "passed": self.passed,
            "threshold_ms": self.threshold_ms,
            "details": self.details,
        }


def _get_memory_usage() -> int:
    """get current process memory in bytes."""
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except ImportError:
        return 0


class E2EBenchmarkSuite:
    """
    end-to-end benchmark suite for realistic workloads.

    all benchmarks:
    - use fixed seeds for reproducibility
    - measure real operations (not mocks)
    - report pass/fail against thresholds
    - are runnable offline
    """

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._results: list[E2EBenchmarkResult] = []
        np.random.seed(seed)

    def _generate_belief(self, idx: int) -> dict[str, Any]:
        """generate a deterministic test belief."""
        return {
            "id": f"belief_{idx:06d}",
            "content": f"test belief content {idx}",
            "confidence": 0.5 + (idx % 100) / 200.0,
            "source": "benchmark",
            "timestamp": 1700000000 + idx,
        }

    def _generate_episode(self, idx: int) -> dict[str, Any]:
        """generate a deterministic test episode."""
        return {
            "id": f"episode_{idx:06d}",
            "beliefs": [f"belief_{i:06d}" for i in range(idx, idx + 3)],
            "action": f"action_{idx % 10}",
            "reward": (idx % 100) / 100.0,
            "timestamp": 1700000000 + idx,
        }

    def bench_belief_insertion(
        self,
        count: int,
        threshold_ms: float,
    ) -> E2EBenchmarkResult:
        """benchmark belief insertion chain."""
        from memory.persistent_memory import PersistentMemory

        memory = PersistentMemory()
        mem_before = _get_memory_usage()

        start = time.perf_counter()
        for i in range(count):
            belief = self._generate_belief(i)
            memory.store_belief(belief)
        duration_ms = (time.perf_counter() - start) * 1000

        mem_after = _get_memory_usage()

        result = E2EBenchmarkResult(
            name=f"belief_insertion_{count}",
            description=f"insert {count} beliefs sequentially",
            duration_ms=duration_ms,
            operations=count,
            ops_per_second=count / (duration_ms / 1000) if duration_ms > 0 else 0,
            memory_delta_bytes=mem_after - mem_before,
            storage_bytes=0,
            seed=self._seed,
            passed=duration_ms <= threshold_ms,
            threshold_ms=threshold_ms,
            details={"final_count": len(memory.list_beliefs())},
        )

        self._results.append(result)
        return result

    def bench_belief_retrieval(
        self,
        count: int,
        threshold_ms: float,
    ) -> E2EBenchmarkResult:
        """benchmark belief retrieval."""
        from memory.persistent_memory import PersistentMemory

        memory = PersistentMemory()

        # setup: insert beliefs
        for i in range(count):
            memory.store_belief(self._generate_belief(i))

        # benchmark: retrieve all
        start = time.perf_counter()
        for i in range(count):
            memory.get_belief(f"belief_{i:06d}")
        duration_ms = (time.perf_counter() - start) * 1000

        result = E2EBenchmarkResult(
            name=f"belief_retrieval_{count}",
            description=f"retrieve {count} beliefs by id",
            duration_ms=duration_ms,
            operations=count,
            ops_per_second=count / (duration_ms / 1000) if duration_ms > 0 else 0,
            memory_delta_bytes=0,
            storage_bytes=0,
            seed=self._seed,
            passed=duration_ms <= threshold_ms,
            threshold_ms=threshold_ms,
        )

        self._results.append(result)
        return result

    def bench_episode_recording(
        self,
        count: int,
        threshold_ms: float,
    ) -> E2EBenchmarkResult:
        """benchmark episode recording with LRU eviction."""
        from memory.episodic_replay import EpisodicReplay

        # use max_episodes = count to avoid eviction during benchmark
        replay = EpisodicReplay(max_episodes=count + 100)
        mem_before = _get_memory_usage()

        start = time.perf_counter()
        for i in range(count):
            episode = self._generate_episode(i)
            replay.record_episode(episode)
        duration_ms = (time.perf_counter() - start) * 1000

        mem_after = _get_memory_usage()

        result = E2EBenchmarkResult(
            name=f"episode_recording_{count}",
            description=f"record {count} episodes",
            duration_ms=duration_ms,
            operations=count,
            ops_per_second=count / (duration_ms / 1000) if duration_ms > 0 else 0,
            memory_delta_bytes=mem_after - mem_before,
            storage_bytes=0,
            seed=self._seed,
            passed=duration_ms <= threshold_ms,
            threshold_ms=threshold_ms,
            details={"final_count": replay.count()},
        )

        self._results.append(result)
        return result

    def bench_episode_replay(
        self,
        total_episodes: int,
        sample_size: int,
        threshold_ms: float,
    ) -> E2EBenchmarkResult:
        """benchmark episode sampling for replay."""
        from memory.episodic_replay import EpisodicReplay

        replay = EpisodicReplay(max_episodes=total_episodes + 100)

        # setup
        for i in range(total_episodes):
            replay.record_episode(self._generate_episode(i))

        # benchmark: sample repeatedly
        iterations = 100
        start = time.perf_counter()
        for i in range(iterations):
            replay.sample(sample_size, seed=self._seed + i)
        duration_ms = (time.perf_counter() - start) * 1000

        result = E2EBenchmarkResult(
            name=f"episode_replay_{total_episodes}_{sample_size}",
            description=f"sample {sample_size} from {total_episodes} episodes x{iterations}",
            duration_ms=duration_ms,
            operations=iterations,
            ops_per_second=iterations / (duration_ms / 1000) if duration_ms > 0 else 0,
            memory_delta_bytes=0,
            storage_bytes=0,
            seed=self._seed,
            passed=duration_ms <= threshold_ms,
            threshold_ms=threshold_ms,
        )

        self._results.append(result)
        return result

    def bench_memory_roundtrip(
        self,
        count: int,
        threshold_ms: float,
    ) -> E2EBenchmarkResult:
        """benchmark save and load of full state."""
        from memory.persistent_memory import PersistentMemory

        memory = PersistentMemory()
        for i in range(count):
            memory.store_belief(self._generate_belief(i))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "beliefs.json"

            # benchmark save
            start_save = time.perf_counter()
            memory.save_to_disk(path)
            save_ms = (time.perf_counter() - start_save) * 1000

            storage_bytes = path.stat().st_size

            # benchmark load
            memory2 = PersistentMemory()
            start_load = time.perf_counter()
            memory2.load_from_disk(path)
            load_ms = (time.perf_counter() - start_load) * 1000

        total_ms = save_ms + load_ms

        result = E2EBenchmarkResult(
            name=f"memory_roundtrip_{count}",
            description=f"save and load {count} beliefs",
            duration_ms=total_ms,
            operations=2,  # save + load
            ops_per_second=2 / (total_ms / 1000) if total_ms > 0 else 0,
            memory_delta_bytes=0,
            storage_bytes=storage_bytes,
            seed=self._seed,
            passed=total_ms <= threshold_ms,
            threshold_ms=threshold_ms,
            details={
                "save_ms": save_ms,
                "load_ms": load_ms,
                "verified_count": len(memory2.list_beliefs()),
            },
        )

        self._results.append(result)
        return result

    def bench_goal_collapse_cycle(
        self,
        goal_count: int,
        threshold_ms: float,
    ) -> E2EBenchmarkResult:
        """benchmark complete goal collapse cycle."""
        from core.goal_collapse import GoalCollapse

        collapse = GoalCollapse()

        # add goals with hierarchy
        collapse.create_goal("root", "root goal", priority=1.0)
        for i in range(goal_count - 1):
            collapse.create_goal(
                goal_id=f"goal_{i}",
                description=f"goal {i}",
                priority=1.0 - (i % 10) * 0.1,
                parent="root",
            )

        # mark some conflicts
        for i in range(min(goal_count // 10, 50)):
            collapse.mark_conflicting(f"goal_{i}", f"goal_{i + 1}")

        # benchmark conflict detection
        start = time.perf_counter()
        conflicts = collapse.find_conflicts()
        for gid in list(collapse._goals.keys())[:100]:
            collapse.get_depth(gid)
        duration_ms = (time.perf_counter() - start) * 1000

        result = E2EBenchmarkResult(
            name=f"goal_collapse_cycle_{goal_count}",
            description=f"goal operations with {goal_count} goals",
            duration_ms=duration_ms,
            operations=goal_count,
            ops_per_second=goal_count / (duration_ms / 1000) if duration_ms > 0 else 0,
            memory_delta_bytes=0,
            storage_bytes=0,
            seed=self._seed,
            passed=duration_ms <= threshold_ms,
            threshold_ms=threshold_ms,
            details={
                "goal_count": len(collapse._goals),
                "conflicts": len(conflicts),
            },
        )

        self._results.append(result)
        return result

    def bench_contradiction_detection(
        self,
        belief_count: int,
        contradiction_count: int,
        threshold_ms: float,
    ) -> E2EBenchmarkResult:
        """benchmark contradiction detection."""
        from core.contradiction_tracer import ContradictionTracer

        tracer = ContradictionTracer()

        # add beliefs
        for i in range(belief_count):
            tracer.register_belief(
                belief_id=f"belief_{i:06d}",
                content=f"content_{i % (belief_count // max(contradiction_count, 1))}",
                confidence=0.5 + (i % 100) / 200.0,
            )

        # mark some contradictions
        for i in range(contradiction_count):
            tracer.mark_contradictory(f"belief_{i:06d}", f"belief_{i + 1:06d}")

        # benchmark detection
        start = time.perf_counter()
        contradictions = tracer.detect_contradictions()
        duration_ms = (time.perf_counter() - start) * 1000

        result = E2EBenchmarkResult(
            name=f"contradiction_detection_{belief_count}",
            description=f"detect contradictions in {belief_count} beliefs",
            duration_ms=duration_ms,
            operations=1,
            ops_per_second=1000 / duration_ms if duration_ms > 0 else 0,
            memory_delta_bytes=0,
            storage_bytes=0,
            seed=self._seed,
            passed=duration_ms <= threshold_ms,
            threshold_ms=threshold_ms,
            details={"contradictions_found": len(contradictions)},
        )

        self._results.append(result)
        return result

    def bench_audit_chain_verification(
        self,
        entry_count: int,
        threshold_ms: float,
    ) -> E2EBenchmarkResult:
        """benchmark audit chain signature verification."""
        from crypto.pq_signatures import (
            generate_keypair, Signer, Verifier, SignedLogChain, Algorithm
        )

        keypair = generate_keypair(Algorithm.ED25519, key_id="bench_key")
        signer = Signer(keypair)
        verifier = Verifier(
            public_key=keypair.public_key,
            algorithm=keypair.algorithm,
        )
        chain = SignedLogChain(signer)

        # build chain
        for i in range(entry_count):
            chain.append({
                "event": f"event_{i}",
                "timestamp": 1700000000 + i,
                "data": f"data_{i}",
            })

        # benchmark verification
        start = time.perf_counter()
        valid, msg = chain.verify_chain(verifier)
        duration_ms = (time.perf_counter() - start) * 1000

        result = E2EBenchmarkResult(
            name=f"audit_chain_verify_{entry_count}",
            description=f"verify {entry_count}-entry audit chain",
            duration_ms=duration_ms,
            operations=entry_count,
            ops_per_second=entry_count / (duration_ms / 1000) if duration_ms > 0 else 0,
            memory_delta_bytes=0,
            storage_bytes=0,
            seed=self._seed,
            passed=duration_ms <= threshold_ms and valid,
            threshold_ms=threshold_ms,
            details={"chain_valid": valid, "message": msg},
        )

        self._results.append(result)
        return result

    def bench_state_transition(
        self,
        transition_count: int,
        threshold_ms: float,
    ) -> E2EBenchmarkResult:
        """benchmark verified state transitions."""
        from verification.state_machine import TransitionEngine, BeliefState

        engine = TransitionEngine()

        # benchmark transitions
        start = time.perf_counter()
        for i in range(transition_count):
            belief = BeliefState(
                belief_id=f"belief_{i:06d}",
                content_hash=f"hash_{i}",
                confidence=0.5,
                timestamp=1700000000.0 + i,
                source="benchmark",
            )
            engine.insert_belief(belief)
        duration_ms = (time.perf_counter() - start) * 1000

        # verify trace
        trace_valid = len(engine.trace) == transition_count

        result = E2EBenchmarkResult(
            name=f"state_transition_{transition_count}",
            description=f"apply {transition_count} verified transitions",
            duration_ms=duration_ms,
            operations=transition_count,
            ops_per_second=transition_count / (duration_ms / 1000) if duration_ms > 0 else 0,
            memory_delta_bytes=0,
            storage_bytes=0,
            seed=self._seed,
            passed=duration_ms <= threshold_ms,
            threshold_ms=threshold_ms,
            details={"trace_length": len(engine.trace), "trace_valid": trace_valid},
        )

        self._results.append(result)
        return result

    def bench_meta_evolution(
        self,
        generations: int,
        threshold_ms: float,
    ) -> E2EBenchmarkResult:
        """benchmark meta-evolution hyperparameter optimization."""
        from core.meta_evolution import (
            MetaEvolutionEngine,
            BeliefCoherenceObjective,
            HyperparameterSet,
        )

        np.random.seed(self._seed)

        objective = BeliefCoherenceObjective()
        engine = MetaEvolutionEngine(
            objective=objective,
            seed=self._seed,
            max_generations=generations,
            convergence_threshold=0.001,
        )

        context = {
            "contradiction_rate": 0.1,
            "stability_score": 0.9,
            "confidence_mean": 0.5,
        }

        start = time.perf_counter()
        result = engine.evolve(context=context)
        duration_ms = (time.perf_counter() - start) * 1000

        bench_result = E2EBenchmarkResult(
            name=f"meta_evolution_{generations}",
            description=f"evolve hyperparameters for {generations} generations",
            duration_ms=duration_ms,
            operations=result.generations,
            ops_per_second=result.generations / (duration_ms / 1000) if duration_ms > 0 else 0,
            memory_delta_bytes=0,
            storage_bytes=0,
            seed=self._seed,
            passed=duration_ms <= threshold_ms,
            threshold_ms=threshold_ms,
            details={
                "converged": result.converged,
                "final_objective": result.final_objective,
                "history_length": len(result.history),
            },
        )

        self._results.append(bench_result)
        return bench_result

    def bench_world_model_simulation(
        self,
        steps: int,
        threshold_ms: float,
    ) -> E2EBenchmarkResult:
        """benchmark world model simulation."""
        from core.world_model import (
            SimpleWorldModel,
            SimulationState,
            SimulationAction,
        )

        np.random.seed(self._seed)

        model = SimpleWorldModel()
        state = SimulationState(
            beliefs={f"b{i}": 0.5 + 0.01 * i for i in range(100)},
            goals={f"g{i}": 0.7 for i in range(10)},
            resources={f"r{i}": 0.8 for i in range(10)},
        )

        actions = [
            SimulationAction(
                action_type="update_belief",
                target_id=f"b{i % 100}",
                delta=0.05,
            )
            for i in range(steps)
        ]

        start = time.perf_counter()
        result = model.simulate(state, actions, seed=self._seed)
        duration_ms = (time.perf_counter() - start) * 1000

        bench_result = E2EBenchmarkResult(
            name=f"world_model_sim_{steps}",
            description=f"simulate {steps} world model steps",
            duration_ms=duration_ms,
            operations=steps,
            ops_per_second=steps / (duration_ms / 1000) if duration_ms > 0 else 0,
            memory_delta_bytes=0,
            storage_bytes=0,
            seed=self._seed,
            passed=duration_ms <= threshold_ms,
            threshold_ms=threshold_ms,
            details={
                "reward": result.reward,
                "deterministic": result.deterministic,
            },
        )

        self._results.append(bench_result)
        return bench_result

    def bench_formal_verification(
        self,
        belief_count: int,
        threshold_ms: float,
    ) -> E2EBenchmarkResult:
        """benchmark formal verification on state."""
        from verification.formal_checker import create_standard_checker

        np.random.seed(self._seed)

        # create state with beliefs
        state = {
            "beliefs": {
                f"b{i}": {"confidence": 0.5 + 0.001 * (i % 100)}
                for i in range(belief_count)
            },
            "goals": {f"g{i}": {"priority": float(i)} for i in range(100)},
            "episodes": [{"timestamp": float(i)} for i in range(100)],
        }

        checker = create_standard_checker(fail_fast=False)

        start = time.perf_counter()
        report = checker.check_all(state)
        duration_ms = (time.perf_counter() - start) * 1000

        bench_result = E2EBenchmarkResult(
            name=f"formal_verify_{belief_count}",
            description=f"verify {belief_count} beliefs with 6 invariants",
            duration_ms=duration_ms,
            operations=report.total_checks,
            ops_per_second=report.total_checks / (duration_ms / 1000) if duration_ms > 0 else 0,
            memory_delta_bytes=0,
            storage_bytes=0,
            seed=self._seed,
            passed=duration_ms <= threshold_ms and report.all_passed,
            threshold_ms=threshold_ms,
            details={
                "invariants_checked": report.total_checks,
                "all_passed": report.all_passed,
            },
        )

        self._results.append(bench_result)
        return bench_result

    def run_standard_suite(self) -> list[E2EBenchmarkResult]:
        """run standard benchmark suite with default thresholds."""
        logger.info("running standard e2e benchmark suite")

        results = [
            self.bench_belief_insertion(1000, threshold_ms=500),
            self.bench_belief_insertion(10000, threshold_ms=5000),
            self.bench_belief_retrieval(10000, threshold_ms=100),
            self.bench_episode_recording(1000, threshold_ms=500),
            self.bench_episode_replay(10000, 100, threshold_ms=500),
            self.bench_memory_roundtrip(10000, threshold_ms=5000),
            self.bench_goal_collapse_cycle(1000, threshold_ms=1000),
            self.bench_contradiction_detection(1000, 10, threshold_ms=500),
            self.bench_audit_chain_verification(1000, threshold_ms=1000),
            self.bench_state_transition(1000, threshold_ms=2000),
            self.bench_meta_evolution(100, threshold_ms=1000),
            self.bench_world_model_simulation(1000, threshold_ms=100),
            self.bench_formal_verification(10000, threshold_ms=50),
        ]

        passed = sum(1 for r in results if r.passed)
        logger.info(f"e2e benchmark suite: {passed}/{len(results)} passed")

        return results

    def get_results(self) -> list[E2EBenchmarkResult]:
        """return all results."""
        return self._results.copy()

    def generate_report(self) -> str:
        """generate text report."""
        lines = [
            "=" * 80,
            "END-TO-END BENCHMARK REPORT",
            f"Seed: {self._seed}",
            "=" * 80,
            "",
        ]

        for r in self._results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"[{status}] {r.name}")
            lines.append(f"      {r.description}")
            lines.append(f"      Duration: {r.duration_ms:.2f}ms (threshold: {r.threshold_ms}ms)")
            lines.append(f"      Ops/sec: {r.ops_per_second:.2f}")
            if r.details:
                for k, v in r.details.items():
                    lines.append(f"      {k}: {v}")
            lines.append("")

        passed = sum(1 for r in self._results if r.passed)
        lines.append(f"Total: {passed}/{len(self._results)} passed")
        lines.append("=" * 80)

        return "\n".join(lines)

    def export_json(self, path: Path | str) -> None:
        """export results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "seed": self._seed,
            "timestamp": time.time(),
            "results": [r.to_dict() for r in self._results],
            "summary": {
                "total": len(self._results),
                "passed": sum(1 for r in self._results if r.passed),
                "failed": sum(1 for r in self._results if not r.passed),
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"exported e2e benchmark results to {path}")


def run_e2e_benchmarks(seed: int = 42) -> list[E2EBenchmarkResult]:
    """run standard e2e benchmark suite."""
    suite = E2EBenchmarkSuite(seed=seed)
    return suite.run_standard_suite()


if __name__ == "__main__":
    suite = E2EBenchmarkSuite(seed=42)
    suite.run_standard_suite()
    print(suite.generate_report())
