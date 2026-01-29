# Threat Models

Per-module threat documentation for Sentinel OS Core.

---

## Overview

This document describes what each security-relevant module defends against, what it does NOT defend against, and recommendations for deployment.

**Philosophy**: No silent downgrades. If a security feature is unavailable, the system fails loudly rather than pretending to be secure.

---

## Isolation Layer (`security/isolation.py`)

### Levels

| Level | What It Defends | What It Does NOT Defend | Requirements |
|-------|-----------------|------------------------|--------------|
| `none` | Nothing | Everything | None |
| `pattern_only` | Accidental dangerous imports, obvious eval/exec | Obfuscation, runtime attacks, determined attackers | None |
| `python_sandbox` | Basic injection, timeout enforcement, restricted builtins | Interpreter exploits, pickle/marshal attacks, ctypes/cffi escape, C extension abuse | None |
| `firejail` | Filesystem access, network (disabled), syscalls (seccomp) | Kernel vulnerabilities, seccomp bypasses, hardware side-channels | `firejail` binary |
| `docker` | Process isolation, filesystem overlay, resource limits | Container escapes, kernel exploits, hypervisor bugs | Docker daemon |

### Threat: Sandbox Escape

**Attack surface**: Python's dynamic nature allows many escape vectors.

**Known escape vectors** (pattern_only and python_sandbox):
- `__builtins__` manipulation
- `pickle.loads()` with malicious payload
- `ctypes` or `cffi` for arbitrary memory access
- Object introspection via `__class__.__mro__`
- `importlib` bypasses

**Mitigation**: Use `firejail` or `docker` for untrusted code. Python-only isolation is defense-in-depth, not a security boundary.

### Threat: Resource Exhaustion

**Attack**: Malicious code runs infinite loops or allocates unbounded memory.

**Defense**: `python_sandbox` enforces timeout. Memory limits require `firejail` cgroups or `docker` resource constraints.

### Stricter Runtime Options (Future)

For higher assurance, consider:

1. **WebAssembly (WASM)**: Compile Python to WASM for hardware-enforced isolation. Tools like `pywasm` or `pyodide` could sandbox execution with no syscall access.

2. **seccomp-bpf directly**: Beyond firejail, a custom seccomp profile can whitelist only safe syscalls (read, write, mmap for existing files).

3. **gVisor**: User-space kernel for container isolation without full VM overhead.

4. **Firecracker microVMs**: Lightweight VMs for maximum isolation with ~125ms boot time.

---

## Privacy Layer (`privacy/`)

### Budget Accountant (`privacy/budget.py`)

**Defends against**: Composition attacks, privacy budget exhaustion, accidental over-spending.

**Does NOT defend against**: Side-channel attacks, timing attacks, reconstruction attacks from auxiliary data.

**Guarantees**:
- ε-δ accounting with hard caps
- No operation proceeds if budget exhausted
- All spending logged

### Mechanisms (`privacy/mechanisms.py`)

| Mechanism | Guarantee | Sensitivity Required | Notes |
|-----------|-----------|---------------------|-------|
| `laplace_mechanism` | ε-differential privacy | Yes | Pure DP |
| `gaussian_mechanism` | (ε,δ)-differential privacy | Yes | Approximate DP |
| `clip_l2` | Bounds sensitivity | N/A | Preprocessing step |

**Threat: Floating-Point Side Channels**

Standard DP implementations can leak information through timing or floating-point precision. This implementation uses `numpy` without constant-time guarantees.

**Mitigation**: For high-security deployments, consider:
- Fixed-point arithmetic
- Constant-time noise sampling
- Hardware RNG

---

## Crypto Layer (`crypto/`)

### ZK Proofs (`crypto/zk_proofs.py`)

**IMPORTANT SCOPE LIMITATION**

The ZK proofs in this system are **discrete-log based only**:

✅ **What we provide**:
- Pedersen commitments (computationally hiding, binding)
- Schnorr proofs of knowledge (prove you know x such that g^x = y)
- State transition proofs (prove belief count deltas)
- Homomorphic commitment addition

❌ **What we do NOT provide**:
- SNARK/STARK (no general computation proofs)
- Range proofs (requires Bulletproofs)
- Set membership proofs (requires accumulators)
- zkEVM or circuit-based proofs

**Why this matters**: If you need to prove arbitrary statements about program execution, these proofs are insufficient. They are suitable for:
- Proving you know a secret without revealing it
- Proving commitment openings
- Proving simple arithmetic relations

**For general-purpose ZK**: Consider integrating:
- `circom` + `snarkjs` for groth16
- `halo2` for PLONKish circuits
- `risc0` for zkVM

### Signatures (`crypto/pq_signatures.py`)

| Algorithm | Status | Notes |
|-----------|--------|-------|
| Ed25519 | Default, always available | ECDSA replacement |
| Dilithium3 | Requires `liboqs-python` | Post-quantum |
| Hybrid Ed25519+Dilithium3 | Requires `liboqs-python` | Defense in depth |

**Threat: Quantum Computers**

Ed25519 is vulnerable to Shor's algorithm. Hybrid mode provides forward secrecy against future quantum attacks.

**Threat: Weak Randomness**

All key generation uses `secrets` module. Ensure system entropy pool is healthy (check `/proc/sys/kernel/random/entropy_avail` on Linux).

### Merkle Trees (`crypto/merkle.py`)

**Defends against**: Proof forgery, selective disclosure attacks.

**Does NOT defend against**: Chosen-prefix attacks on underlying hash (SHA-256 is resistant).

### Homomorphic Encryption (`crypto/homomorphic.py`)

**Requires**: `tenseal` library.

**Threat model**:
- CKKS scheme for approximate arithmetic
- Suitable for encrypted aggregation, not exact computation
- Rounding errors accumulate over operations

**Does NOT provide**: Bootstrapping (limited multiplication depth), exact integer arithmetic.

---

## Audit Logger (`security/audit_logger.py`)

**Defends against**: Log tampering, repudiation.

**Does NOT defend against**: Log deletion (requires append-only storage), timing attacks.

**Guarantees**:
- HMAC on every entry
- Chain integrity verification
- Signed log chains with Ed25519

---

## World Model (`core/world_model.py`)

**Security relevance**: Counterfactual simulations could be used to probe system behavior.

**Threat**: Adversarial inputs designed to cause expensive simulations (DoS).

**Mitigation**: Max step limits enforced. Simulation timeouts configurable.

---

## Recommendations by Deployment Scenario

### Development/Testing
- Use `pattern_only` or `python_sandbox`
- No external dependencies required

### Internal Deployment (Trusted Environment)
- Use `python_sandbox` with strict builtins
- Enable HMAC audit logging
- Monitor privacy budget

### External/Hostile Environment
- Use `firejail` or `docker`
- Enable hybrid PQ signatures if `liboqs` available
- Deploy in isolated VM or container
- Consider external pentest before production

### Air-Gapped/Offline
- Verify all dependencies are bundled
- Use deterministic builds
- Enable full audit logging
- No network isolation needed (already offline)

---

## Bug Bounty / Security Contact

For security vulnerabilities:
1. Open a private security advisory on GitHub
2. Email maintainers directly (see CONTRIBUTING.md)
3. Do NOT open public issues for security bugs

We aim to respond within 48 hours and fix critical issues within 7 days.

---

*Last updated: 2026-01-29*
