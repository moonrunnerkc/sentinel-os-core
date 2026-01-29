# Author: Bradley R. Kinnard
# extended benchmarks - 1000+ episodes, long-running stability testing

import gc
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from utils.helpers import get_logger

logger = get_logger(__name__)


def _get_memory_mb() -> float:
    """get current process memory in megabytes."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


@dataclass
class EpisodeMetrics:
    """metrics for a single episode cycle."""
    episode_num: int
    beliefs_created: int
    beliefs_decayed: int
    contradictions_found: int
    contradictions_resolved: int
    goals_evolved: int
    goal_convergence_rate: float
    cycle_time_ms: float
    memory_mb: float
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExtendedBenchmarkResult:
    """full result of an extended benchmark run."""
    name: str
    total_episodes: int
    completed_episodes: int
    total_runtime_seconds: float
    avg_cycle_time_ms: float
    p95_cycle_time_ms: float
    p99_cycle_time_ms: float
    peak_memory_mb: float
    final_memory_mb: float
    total_beliefs_created: int
    total_beliefs_decayed: int
    total_contradictions_found: int
    total_contradictions_resolved: int
    total_goals_evolved: int
    avg_convergence_rate: float
    total_errors: int
    error_rate_percent: float
    seed: int
    timestamp: str
    episode_metrics: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class ExtendedBenchmarkSuite:
    """
    long-running stability benchmark.

    runs 1000+ episodes of the belief-update-goal-evolve cycle,
    collecting metrics on every iteration. outputs raw logs for
    external analysis.
    """

    def __init__(self, seed: int = 42, output_dir: str = "data/logs"):
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _generate_belief_content(self, idx: int) -> str:
        """generate diverse belief content."""
        templates = [
            "observation about entity_{0}",
            "hypothesis regarding process_{0}",
            "fact: measurement_{0} recorded",
            "inference from context_{0}",
            "assumption about state_{0}",
        ]
        return templates[idx % len(templates)].format(idx)

    def run_extended_benchmark(
        self,
        num_episodes: int = 1000,
        beliefs_per_episode: int = 10,
        log_every: int = 100,
    ) -> ExtendedBenchmarkResult:
        """
        run extended stability benchmark.

        each episode:
        1. create beliefs
        2. link some beliefs
        3. propagate confidence
        4. apply decay
        5. detect contradictions
        6. resolve contradictions
        7. evolve goals
        8. collect metrics
        """
        from core.belief_ecology import BeliefEcology
        from core.goal_collapse import GoalCollapse
        from core.contradiction_tracer import ContradictionTracer

        logger.info(f"starting extended benchmark: {num_episodes} episodes, seed={self._seed}")

        np.random.seed(self._seed)
        ecology = BeliefEcology()
        goal_engine = GoalCollapse()
        tracer = ContradictionTracer()

        # setup root goals
        goal_engine.create_goal("root", "main objective", priority=1.0)
        goal_engine.create_goal("sub1", "subgoal 1", priority=0.7, parent="root")
        goal_engine.create_goal("sub2", "subgoal 2", priority=0.5, parent="root")

        episode_metrics: list[EpisodeMetrics] = []
        cycle_times: list[float] = []
        total_start = time.perf_counter()
        peak_memory = 0.0

        for ep in range(num_episodes):
            cycle_start = time.perf_counter()
            errors: list[str] = []

            # 1. create beliefs
            beliefs_created = 0
            for i in range(beliefs_per_episode):
                b_id = f"b_{ep:05d}_{i:03d}"
                try:
                    ecology.create_belief(
                        belief_id=b_id,
                        content=self._generate_belief_content(ep * beliefs_per_episode + i),
                        confidence=float(self._rng.uniform(0.3, 0.9)),
                        source="benchmark"
                    )
                    beliefs_created += 1
                except Exception as e:
                    errors.append(f"create_belief: {e}")

            # 2. link some beliefs (random pairs within episode)
            if beliefs_created >= 2:
                for _ in range(min(5, beliefs_created // 2)):
                    src_idx = self._rng.integers(0, beliefs_created)
                    tgt_idx = self._rng.integers(0, beliefs_created)
                    if src_idx != tgt_idx:
                        src_id = f"b_{ep:05d}_{src_idx:03d}"
                        tgt_id = f"b_{ep:05d}_{tgt_idx:03d}"
                        try:
                            ecology.link_beliefs(src_id, tgt_id, float(self._rng.uniform(0.2, 0.8)))
                        except Exception as e:
                            errors.append(f"link_beliefs: {e}")

            # 3. propagate from random belief
            if beliefs_created > 0:
                src_id = f"b_{ep:05d}_{self._rng.integers(0, beliefs_created):03d}"
                try:
                    ecology.propagate(src_id)
                except Exception as e:
                    errors.append(f"propagate: {e}")

            # 4. apply decay to older beliefs
            beliefs_decayed = 0
            if ep > 0:
                decay_ep = self._rng.integers(0, ep)
                for i in range(min(3, beliefs_per_episode)):
                    old_id = f"b_{decay_ep:05d}_{i:03d}"
                    try:
                        ecology.apply_decay(old_id, decay_rate=0.02)
                        beliefs_decayed += 1
                    except KeyError:
                        pass  # belief may not exist
                    except Exception as e:
                        errors.append(f"apply_decay: {e}")

            # 5. detect contradictions (simulate some)
            contradictions_found = 0
            if beliefs_created >= 2 and self._rng.random() < 0.1:
                # 10% chance of contradiction per episode
                a_idx = self._rng.integers(0, beliefs_created)
                b_idx = self._rng.integers(0, beliefs_created)
                if a_idx != b_idx:
                    a_id = f"b_{ep:05d}_{a_idx:03d}"
                    b_id = f"b_{ep:05d}_{b_idx:03d}"
                    try:
                        ecology.mark_contradictory(a_id, b_id)
                        contradictions_found += 1
                    except Exception as e:
                        errors.append(f"mark_contradictory: {e}")

            # 6. resolve contradictions (reduce confidence of one)
            contradictions_resolved = 0
            all_contradictions = ecology.find_contradictions()
            if all_contradictions and self._rng.random() < 0.3:
                # resolve 30% of the time
                try:
                    pair = all_contradictions[self._rng.integers(0, len(all_contradictions))]
                    # reduce confidence of lower-priority belief
                    b1 = ecology.get_belief(pair[0])
                    b2 = ecology.get_belief(pair[1])
                    loser = pair[0] if b1["confidence"] < b2["confidence"] else pair[1]
                    ecology.apply_decay(loser, decay_rate=0.5)
                    contradictions_resolved += 1
                except Exception as e:
                    errors.append(f"resolve_contradiction: {e}")

            # 7. evolve goals
            goals_evolved = 0
            convergence_rates: list[float] = []
            for g_id in ["sub1", "sub2"]:
                try:
                    converged, iters = goal_engine.evolve_until_stable(
                        g_id,
                        epsilon=1e-4,
                        max_iterations=50,
                        seed=self._seed + ep
                    )
                    goals_evolved += 1
                    convergence_rates.append(1.0 if converged else iters / 50.0)
                except Exception as e:
                    errors.append(f"evolve_goal: {e}")
                    convergence_rates.append(0.0)

            # metrics
            cycle_time = (time.perf_counter() - cycle_start) * 1000
            cycle_times.append(cycle_time)
            mem_mb = _get_memory_mb()
            peak_memory = max(peak_memory, mem_mb)

            metrics = EpisodeMetrics(
                episode_num=ep,
                beliefs_created=beliefs_created,
                beliefs_decayed=beliefs_decayed,
                contradictions_found=contradictions_found,
                contradictions_resolved=contradictions_resolved,
                goals_evolved=goals_evolved,
                goal_convergence_rate=np.mean(convergence_rates) if convergence_rates else 0.0,
                cycle_time_ms=cycle_time,
                memory_mb=mem_mb,
                errors=errors,
            )
            episode_metrics.append(metrics)

            if (ep + 1) % log_every == 0:
                avg_recent = np.mean(cycle_times[-log_every:])
                logger.info(
                    f"episode {ep+1}/{num_episodes} | "
                    f"avg cycle: {avg_recent:.2f}ms | "
                    f"memory: {mem_mb:.1f}MB | "
                    f"beliefs: {ecology.count()}"
                )

        total_runtime = time.perf_counter() - total_start
        final_memory = _get_memory_mb()

        # aggregate
        cycle_times_arr = np.array(cycle_times)
        total_errors = sum(len(m.errors) for m in episode_metrics)

        result = ExtendedBenchmarkResult(
            name="extended_belief_goal_cycle",
            total_episodes=num_episodes,
            completed_episodes=len(episode_metrics),
            total_runtime_seconds=total_runtime,
            avg_cycle_time_ms=float(np.mean(cycle_times_arr)),
            p95_cycle_time_ms=float(np.percentile(cycle_times_arr, 95)),
            p99_cycle_time_ms=float(np.percentile(cycle_times_arr, 99)),
            peak_memory_mb=peak_memory,
            final_memory_mb=final_memory,
            total_beliefs_created=sum(m.beliefs_created for m in episode_metrics),
            total_beliefs_decayed=sum(m.beliefs_decayed for m in episode_metrics),
            total_contradictions_found=sum(m.contradictions_found for m in episode_metrics),
            total_contradictions_resolved=sum(m.contradictions_resolved for m in episode_metrics),
            total_goals_evolved=sum(m.goals_evolved for m in episode_metrics),
            avg_convergence_rate=float(np.mean([m.goal_convergence_rate for m in episode_metrics])),
            total_errors=total_errors,
            error_rate_percent=(total_errors / num_episodes) * 100 if num_episodes > 0 else 0,
            seed=self._seed,
            timestamp=datetime.utcnow().isoformat() + "Z",
            episode_metrics=[m.to_dict() for m in episode_metrics],
        )

        return result

    def save_raw_logs(self, result: ExtendedBenchmarkResult) -> Path:
        """save full benchmark result as JSON for external analysis."""
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"extended_benchmark_{ts}.json"
        filepath = self._output_dir / filename

        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"raw logs saved to: {filepath}")
        return filepath

    def print_summary(self, result: ExtendedBenchmarkResult) -> None:
        """print human-readable summary."""
        print("\n" + "=" * 70)
        print("EXTENDED BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"Benchmark:       {result.name}")
        print(f"Episodes:        {result.completed_episodes}/{result.total_episodes}")
        print(f"Total Runtime:   {result.total_runtime_seconds:.2f}s")
        print(f"Seed:            {result.seed}")
        print(f"Timestamp:       {result.timestamp}")
        print()
        print("TIMING")
        print(f"  Avg Cycle:     {result.avg_cycle_time_ms:.2f}ms")
        print(f"  P95 Cycle:     {result.p95_cycle_time_ms:.2f}ms")
        print(f"  P99 Cycle:     {result.p99_cycle_time_ms:.2f}ms")
        print()
        print("MEMORY")
        print(f"  Peak:          {result.peak_memory_mb:.1f}MB")
        print(f"  Final:         {result.final_memory_mb:.1f}MB")
        print()
        print("BELIEFS")
        print(f"  Created:       {result.total_beliefs_created}")
        print(f"  Decayed:       {result.total_beliefs_decayed}")
        print()
        print("CONTRADICTIONS")
        print(f"  Found:         {result.total_contradictions_found}")
        print(f"  Resolved:      {result.total_contradictions_resolved}")
        print()
        print("GOALS")
        print(f"  Evolved:       {result.total_goals_evolved}")
        print(f"  Avg Converge:  {result.avg_convergence_rate:.2%}")
        print()
        print("ERRORS")
        print(f"  Total:         {result.total_errors}")
        print(f"  Error Rate:    {result.error_rate_percent:.2f}%")
        print("=" * 70)


def main():
    """run extended benchmark from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run extended stability benchmark")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--beliefs", type=int, default=10, help="Beliefs per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-every", type=int, default=100, help="Log progress every N episodes")
    parser.add_argument("--output", type=str, default="data/logs", help="Output directory")
    args = parser.parse_args()

    suite = ExtendedBenchmarkSuite(seed=args.seed, output_dir=args.output)
    result = suite.run_extended_benchmark(
        num_episodes=args.episodes,
        beliefs_per_episode=args.beliefs,
        log_every=args.log_every,
    )
    suite.print_summary(result)
    log_path = suite.save_raw_logs(result)
    print(f"\nRaw logs: {log_path}")


if __name__ == "__main__":
    main()
