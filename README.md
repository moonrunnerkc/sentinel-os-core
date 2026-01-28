# Sentinel OS Core

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

Modular, offline-first cognitive operating system for synthetic intelligence. Designed for autonomous reasoning, persistent memory, and goal evolution in air-gapped or adversarial environments.

**Author:** Bradley R. Kinnard

---

## Table of Contents

- [Why This Exists](#why-this-exists)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [CLI Usage](#cli-usage)
- [Programmatic Usage](#programmatic-usage)
- [Testing](#testing)
- [Docker](#docker)
- [Security Notes](#security-notes)
- [Limitations](#limitations)
- [License](#license)

---

## Why This Exists

Most AI systems assume cloud connectivity and treat security as an afterthought. Sentinel OS Core is built for scenarios where:

- Network access is unavailable or untrusted
- Code execution must be sandboxed and auditable
- Beliefs and goals must evolve deterministically
- All state changes must be traceable

This is not a chatbot framework. It is a cognitive substrate for autonomous systems.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SENTINEL OS CORE                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   INPUT     │  │   OUTPUT    │  │     LOCAL LLM           │  │
│  │   LAYER     │──│   LAYER     │──│   (llama.cpp)           │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
│         │                │                     │                │
│  ┌──────▼────────────────▼─────────────────────▼──────────────┐ │
│  │                    CORE ENGINE                              │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │ │
│  │  │ BeliefEcology│  │ GoalCollapse │  │ContradictionTracer│ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘  │ │
│  └────────────────────────────┬───────────────────────────────┘ │
│                               │                                 │
│  ┌────────────────────────────▼───────────────────────────────┐ │
│  │                    MEMORY SYSTEM                            │ │
│  │  ┌──────────────────┐  ┌────────────────────────────────┐  │ │
│  │  │ PersistentMemory │  │ EpisodicReplay (LRU eviction)  │  │ │
│  │  └──────────────────┘  └────────────────────────────────┘  │ │
│  └────────────────────────────┬───────────────────────────────┘ │
│                               │                                 │
│  ┌────────────────────────────▼───────────────────────────────┐ │
│  │                   SECURITY LAYER                            │ │
│  │  ┌──────────────┐  ┌────────────────────────────────────┐  │ │
│  │  │   Sandbox    │  │ AuditLogger (HMAC-signed)          │  │ │
│  │  └──────────────┘  └────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

### Core (Implemented)

| Feature | Description | Location |
|---------|-------------|----------|
| **Belief Ecology** | Dynamic belief network with propagation and decay | [core/belief_ecology.py](core/belief_ecology.py) |
| **Goal Collapse** | RL-based goal evolution with Laplace noise for differential privacy | [core/goal_collapse.py](core/goal_collapse.py) |
| **Contradiction Tracing** | Automatic detection and resolution of conflicting beliefs | [core/contradiction_tracer.py](core/contradiction_tracer.py) |
| **Persistent Memory** | Async I/O via aiofiles for belief/episode storage | [memory/persistent_memory.py](memory/persistent_memory.py) |
| **Episodic Replay** | LRU-based episode storage with deterministic sampling | [memory/episodic_replay.py](memory/episodic_replay.py) |
| **Sandbox Execution** | Restricted builtins, blocked imports (os, subprocess, eval, exec) | [security/sandbox.py](security/sandbox.py) |
| **HMAC Audit Logs** | Tamper-evident logging with HKDF-derived session keys | [security/audit_logger.py](security/audit_logger.py) |
| **Input Validation** | Blocks common prompt injection patterns | [interfaces/input_layer.py](interfaces/input_layer.py) |
| **D3.js Visualizer** | Browser-based graph visualization for beliefs/goals | [graphs/visualizer.html](graphs/visualizer.html) |
| **Local LLM Interface** | llama-cpp-python integration with GPU auto-detection | [interfaces/local_llm.py](interfaces/local_llm.py) |

### Experimental (Config-Gated)

These features are disabled by default. Enable via config flags. Some are stubs or mocks.

| Feature | Config Flag | Status | Notes |
|---------|-------------|--------|-------|
| Homomorphic Encryption | `use_homomorphic_enc` | Stub | Imports TenSEAL if available, falls back gracefully |
| Neuromorphic Mode | `neuromorphic_mode` | Stub | Imports brian2 if available, no functional SNN yet |
| Firejail Sandbox | `use_firejail` | Optional | Checks for firejail binary, uses seccomp profile |
| Post-Quantum Crypto | `pq_crypto` | Placeholder | Falls back to HKDF; Kyber/Dilithium not implemented |
| Federated Sync | `enable_federated_sync` | Mock | Simulated ZKP proofs using random bytes |
| World Models | `use_world_models` | Stub | Config flag exists, no physics simulation |

---

## Installation

```bash
git clone https://github.com/moonrunnerkc/sentinel-os-core.git
cd sentinel-os-core
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dependencies

Core dependencies from [requirements.txt](requirements.txt):

- numpy, scipy, pyyaml, jsonschema, aiofiles
- pytest, pytest-cov, pytest-timeout, pytest-benchmark
- llama-cpp-python (optional, for LLM inference)
- cryptography (for HMAC/HKDF)
- flake8, mypy, bandit (dev tools)

Optional (for experimental features):

- tenseal (homomorphic encryption)
- brian2 (neuromorphic simulation)
- mujoco (physics simulation)

---

## Quick Start

```python
from main import SentinelOS

os = SentinelOS()
os.start()

# create a belief
os.process_input({
    "type": "belief",
    "content": "the system is stable",
    "priority": 0.9
})

# create a goal
os.process_input({
    "type": "goal",
    "content": "maintain stability",
    "priority": 1.0
})

print(os.get_status())
os.stop()
```

---

## Configuration

### System Config

Edit [config/system_config.yaml](config/system_config.yaml):

```yaml
llm:
  backend: llama-cpp
  model_path: data/models/your-model.gguf
  temperature: 0.0
  seed: 42
  gpu_layers: 0

performance:
  max_beliefs: 10000
  max_episodes: 10000
  cache_size: 1000

features:
  use_world_models: false
  neuromorphic_mode: false
  enable_meta_evolution: false
```

### Security Config

Edit [config/security_rules.json](config/security_rules.json):

```json
{
  "use_firejail": false,
  "allowed_paths": ["data/"],
  "hmac_key_seed": 42,
  "seccomp_profile": "execve,ptrace",
  "pq_crypto": false,
  "use_homomorphic_enc": false,
  "enable_federated_sync": false
}
```

---

## CLI Usage

```bash
python main.py --config config/system_config.yaml
```

---

## Programmatic Usage

```python
from main import SentinelOS

# initialize with custom paths
os = SentinelOS(
    config_path="config/system_config.yaml",
    security_path="config/security_rules.json"
)

os.start()

# process inputs
result = os.process_input({
    "type": "belief",
    "content": "system is operational",
    "priority": 0.8
})

# get system status
status = os.get_status()
print(f"Beliefs: {status['beliefs']}, Goals: {status['goals']}")

# export introspection graph
os.export_graph("data/graphs/snapshot.json")

os.stop()
```

---

## Testing

### Run Tests

```bash
# fast tests (recommended for development)
pytest tests/ -m "not slow and not chaos" -v

# all tests except chaos
pytest tests/ -m "not chaos" -v

# chaos engineering tests (fault injection)
pytest tests/ -m chaos -v

# with coverage report
pytest tests/ --cov=core --cov=memory --cov=security --cov=interfaces --cov-report=term-missing
```

### Run Benchmarks

```bash
pytest tests/benchmarks.py -v --benchmark-only
```

### Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.slow` | Large-scale tests (10k+ items) |
| `@pytest.mark.chaos` | Fault injection tests |

---

## Docker

### Build and Run

```bash
docker build -t sentinel-os .
docker run --network none sentinel-os
```

### Docker Compose

```bash
docker-compose up
```

The compose file runs with `network_mode: none` for offline isolation.

---

## Security Notes

### What Is Implemented

- **HMAC-signed audit logs**: Each log entry includes an HMAC computed with a session-derived key. Verification detects tampering. See [security/audit_logger.py](security/audit_logger.py).

- **Sandbox execution**: Blocks dangerous builtins (`__import__`, `eval`, `exec`, `compile`, `open`). Uses restricted globals. See [security/sandbox.py](security/sandbox.py).

- **Differential privacy**: Goal rewards include Laplace noise (epsilon=0.1 default). See [core/goal_collapse.py](core/goal_collapse.py).

- **Input validation**: Blocks common prompt injection patterns. See [interfaces/input_layer.py](interfaces/input_layer.py).

### What Is NOT Implemented

- **Post-quantum cryptography**: The `pq_crypto` flag exists but falls back to SHA256-based HKDF. Kyber/Dilithium are not integrated.

- **True homomorphic encryption**: TenSEAL is in requirements but the implementation is a stub that catches ImportError.

- **Kernel-level isolation**: The Python sandbox is best-effort. For adversarial code, use VMs or hardware isolation.

- **ZKP verification**: Federated sync uses random bytes as mock proofs. Not cryptographically valid.

### Recommendations

For production use in adversarial environments:

1. Run inside a dedicated VM or container with no network
2. Enable firejail if available (`use_firejail: true`)
3. Do not trust the sandbox against malicious code
4. Treat experimental features as research-only

---

## Limitations

| Limitation | Details |
|------------|---------|
| **Sandbox escapes** | Python-based isolation is bypassable. Not a security boundary against determined attackers. |
| **LLM reproducibility** | Outputs may vary across hardware due to quantization and floating-point differences. |
| **Scalability** | Tested to 10k beliefs/episodes. Larger scales require profiling. |
| **No CI/CD** | No GitHub Actions workflows. Tests run locally. |
| **Experimental features** | Stubs only. Do not rely on them for production. |

---

## License

MIT License. See [LICENSE](LICENSE).

---

*Built for systems that must think alone.*
