# Changelog

All notable changes to Sentinel OS Core. Follows [Keep a Changelog](https://keepachangelog.com/).

---

## [Unreleased]

### Added
- Extended 1000+ episode benchmarks with raw log export
- Full belief-update-goal-evolve cycle demo
- Explicit per-module threat model documentation
- CONTRIBUTING.md with guidelines
- Roadmap section in README
- Coverage report badges

### Changed
- Quantified efficiency metrics in meta-evolution objectives
- Clearer ZK scope limitations upfront

---

## [0.1.0-alpha] - 2026-01-29

### Added
- **Core Engine**: BeliefEcology, GoalCollapse, ContradictionTracer
- **Verification Layer**: StateMachine, Invariants, PropertyTests, FormalChecker
- **Privacy Layer**: BudgetAccountant (ε,δ tracking), Laplace/Gaussian mechanisms
- **Crypto Layer**:
  - Pedersen commitments + Schnorr proofs (discrete log)
  - Ed25519 signatures with hybrid PQ mode (liboqs optional)
  - Merkle trees for batch verification
  - Signed audit chains
- **Memory**: PersistentMemory (aiofiles), EpisodicReplay (LRU eviction)
- **Isolation**: Explicit levels (none, pattern_only, python_sandbox, firejail, docker)
- **Meta-Evolution**: Hyperparameter optimization with measurable objectives
- **World Model**: Simple causal simulation, counterfactual analysis
- **Authenticated Sync**: Signed belief export/import with replay protection

### Architecture Decisions
- Offline-first design: no cloud dependencies
- Hard-fail semantics: isolation levels never silently downgrade
- Deterministic replay: all RL operations seeded and logged
- Defense in depth: multiple isolation layers documented with threat models

### Tests
- 360 tests passing, 11 skipped (external deps)
- Property testing via Hypothesis
- Scalability validated to 10k beliefs

### Known Limitations
- Python sandbox is bypassable (documented)
- ZK proofs are discrete-log only (NOT SNARK/STARK)
- Neuromorphic mode requires brian2, no mock fallback

---

## [0.0.1] - 2026-01-15

### Added
- Initial project structure
- Basic belief ecology prototype
- Goal collapse RL skeleton
- HMAC audit logging

---

## Git History Summary

Key milestones (condensed from `git log --oneline`):

```
2026-01-29 Extended benchmarks, CONTRIBUTING, threat model docs
2026-01-28 ZK proofs (Pedersen/Schnorr), authenticated sync
2026-01-27 Meta-evolution engine, world model integration
2026-01-26 Privacy layer (DP mechanisms, budget accountant)
2026-01-25 Isolation module with explicit levels, hard-fail
2026-01-24 Formal verification layer (state machine, invariants)
2026-01-22 Crypto layer (commitments, signatures, Merkle trees)
2026-01-20 Persistent memory, episodic replay with LRU
2026-01-18 Core engine (belief ecology, goal collapse, contradiction tracer)
2026-01-15 Project initialization, structure, requirements
```

---

## Alpha Testing Instructions

**This is an alpha release.** Expect sharp edges.

### Recommended Setup
1. Clone into an isolated VM or container
2. Run all tests: `pytest tests/ -v`
3. Run benchmarks: `python -m benchmarks.e2e_benchmark`
4. Run extended benchmark: `python -m benchmarks.extended_benchmark`

### Feedback
Open issues for bugs, missing threat coverage, or integration blockers.

---

*Format: [version] - YYYY-MM-DD*
