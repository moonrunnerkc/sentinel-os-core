# Author: Bradley R. Kinnard
# benchmarks for performance testing

import pytest
import time


@pytest.mark.benchmark
def test_belief_throughput(benchmark):
    """benchmark belief processing throughput."""
    from core import BeliefEcology

    ecology = BeliefEcology()

    def process_beliefs():
        for i in range(1000):
            ecology.create_belief(f"b{i}", f"belief {i}", 0.5, "test")
        return len(ecology._beliefs)

    result = benchmark(process_beliefs)
    # target: <50ms for 1k beliefs (we benchmark the function, not assert timing)


@pytest.mark.benchmark
def test_goal_collapse_latency(benchmark):
    """benchmark goal collapse convergence time."""
    from core import GoalCollapse

    collapse = GoalCollapse()
    collapse.create_goal("bench", "benchmark goal", 0.5)

    def evolve():
        collapse.reset_goal("bench", 0.5)
        converged, iters = collapse.evolve_until_stable(
            "bench", epsilon=1e-3, max_iterations=100, seed=42
        )
        return iters

    result = benchmark(evolve)
    # target: <100ms


@pytest.mark.benchmark
def test_memory_write_throughput(benchmark):
    """benchmark memory write performance."""
    from memory import PersistentMemory

    memory = PersistentMemory()

    def write_beliefs():
        for i in range(1000):
            memory.store_belief({"id": f"m{i}", "content": f"mem {i}", "confidence": 0.5})
        return len(memory._beliefs)

    result = benchmark(write_beliefs)


@pytest.mark.benchmark
def test_episodic_replay_sampling(benchmark):
    """benchmark episode sampling performance."""
    from memory import EpisodicReplay

    replay = EpisodicReplay(max_episodes=10000)

    # pre-populate
    for i in range(5000):
        replay.record_episode({"id": f"e{i}", "data": f"episode {i}"})

    def sample():
        return replay.sample(100, seed=42)

    result = benchmark(sample)


@pytest.mark.benchmark
def test_sandbox_execution(benchmark):
    """benchmark sandbox execution overhead."""
    from security import Sandbox

    sandbox = Sandbox()

    def execute():
        return sandbox.execute_safely("x = 1 + 1")

    result = benchmark(execute)


@pytest.mark.benchmark
def test_audit_log_write(benchmark):
    """benchmark audit log write with HMAC."""
    from security import AuditLogger

    audit = AuditLogger()

    def log_entry():
        return audit.log("benchmark_event", "INFO", {"iteration": 1})

    result = benchmark(log_entry)


@pytest.mark.benchmark
def test_graph_construction(benchmark):
    """benchmark introspection graph construction."""
    from graphs import IntrospectionGraph

    def build_graph():
        graph = IntrospectionGraph()
        for i in range(100):
            graph.add_node(f"n{i}", "belief", f"node {i}")
        for i in range(50):
            graph.add_edge(f"n{i}", f"n{i+50}", "link")
        return graph.node_count()

    result = benchmark(build_graph)


@pytest.mark.benchmark
def test_input_validation(benchmark):
    """benchmark input validation throughput."""
    from interfaces import InputLayer

    layer = InputLayer()
    input_data = {"type": "belief", "content": "test content for validation"}

    def validate():
        return layer.process(input_data)

    result = benchmark(validate)


@pytest.mark.benchmark
def test_federated_proof_generation(benchmark):
    """benchmark ZKP proof generation."""
    from interfaces import FederatedSync

    sync = FederatedSync(enabled=True)

    def generate():
        return sync.generate_proof("test_belief", "secret content", 0.8)

    result = benchmark(generate)
    # target: <50ms per proof


@pytest.mark.benchmark
def test_contradiction_detection(benchmark):
    """benchmark contradiction detection."""
    from core import ContradictionTracer

    tracer = ContradictionTracer()

    # pre-populate
    for i in range(100):
        tracer.register_belief(f"b{i}", f"belief {i}")
    for i in range(0, 50, 2):
        tracer.mark_contradictory(f"b{i}", f"b{i+1}")

    def detect():
        return tracer.detect_contradictions()

    result = benchmark(detect)
