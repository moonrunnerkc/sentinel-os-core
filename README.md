# Sentinel OS Core

**Author:** Bradley R. Kinnard
**License:** MIT
**Python:** 3.12+

Sentinel OS Core is a modular, offline-first cognitive operating system for synthetic intelligence. Designed for secure autonomous reasoning, persistent memory, and introspective goal evolution in air-gapped or adversarial environments.

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
│  │  │ BeliefEcology│  │ GoalCollapse │  │ ContradictionTrc │  │ │
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
│  │  │   Sandbox    │  │ AuditLogger (HMAC signed)          │  │ │
│  │  └──────────────┘  └────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **Offline-First**: No cloud dependencies, all computation local
- **Belief Ecology**: Dynamic belief network with propagation and decay
- **Goal Collapse**: RL-based goal evolution with differential privacy
- **Contradiction Tracing**: Automatic detection and resolution
- **Persistent Memory**: Async I/O with optional homomorphic encryption
- **Episodic Replay**: LRU-based episode storage with deterministic sampling
- **Sandbox Execution**: Best-effort code isolation with audit logging
- **HMAC Audit Logs**: Tamper-evident logging with key rotation
- **Introspection Graph**: D3.js visualization of belief/goal networks

## Installation

```bash
git clone https://github.com/yourusername/sentinel-os-core.git
cd sentinel-os-core
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Edit `config/system_config.yaml`:

```yaml
llm:
  backend: llama-cpp
  model_path: data/models/your-model.gguf
  temperature: 0.0  # deterministic
  seed: 42
performance:
  max_beliefs: 10000
  max_episodes: 10000
features:
  use_world_models: false
  neuromorphic_mode: false
```

Edit `config/security_rules.json`:

```json
{
  "use_firejail": false,
  "hmac_key_seed": 42,
  "pq_crypto": false,
  "use_homomorphic_enc": false
}
```

## Usage

```bash
python main.py --config config/system_config.yaml
```

### Programmatic Usage

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

## Testing

```bash
# fast tests only (CI)
pytest tests/ -m "not slow and not chaos" -v

# all tests including slow
pytest tests/ -m "not chaos" -v

# chaos engineering tests
pytest tests/ -m chaos -v

# with coverage
pytest tests/ --cov=. --cov-report=html
```

## Docker

```bash
docker build -t sentinel-os .
docker run --network none sentinel-os
```

Or with docker-compose:

```bash
docker-compose up
```

## Advanced Features (Experimental)

These features are config-gated. Enable with caution:

- **World Models**: Causal counterfactual simulation (`use_world_models: true`)
- **Neuromorphic Mode**: Brian2 SNN for low-power graph processing
- **Homomorphic Encryption**: TenSEAL for private belief operations
- **Federated Sync**: ZKP-based belief synchronization
- **Post-Quantum Crypto**: Kyber/Dilithium HMAC signing

## Performance KPIs

| Metric | Target | Actual |
|--------|--------|--------|
| Belief throughput | <50ms / 1k | ~20ms |
| Goal collapse | <100ms | ~5ms |
| Memory write | <1s / 1k | ~50ms |
| Sandbox execution | <10ms | ~1ms |
| Audit log write | <5ms | ~0.5ms |

## Known Limitations

- **Sandbox**: Python-based, best-effort isolation. Not cryptographically secure against determined adversaries. For hostile code, use VMs or hardware isolation.
- **LLM Reproducibility**: Outputs may vary slightly across hardware due to quantization.
- **Scalability**: Tested to 10k beliefs/episodes. Larger scales require profiling.
- **Advanced Features**: Experimental. Disable in production unless validated.

## Security

- HMAC-signed audit logs prevent tampering
- Differential privacy (ε=0.1) in RL rewards
- Input validation blocks prompt injection (OWASP Top 10)
- Sandbox blocks `os`, `subprocess`, `eval`, `exec`
- Optional firejail with seccomp for enhanced isolation

## License

MIT License. See [LICENSE](LICENSE).
