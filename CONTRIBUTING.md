# Contributing to Sentinel OS Core

Thanks for your interest in contributing! This document covers how to get started.

---

## Quick Start

```bash
# fork and clone
git clone https://github.com/moonrunnerkc/sentinel-os-core.git
cd sentinel-os-core

# set up environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# run tests
pytest tests/ -v

# run demo
python demo.py
```

---

## Development Workflow

1. **Fork** the repository
2. **Create a branch** for your feature: `git checkout -b feature/my-feature`
3. **Write tests first** - we use TDD
4. **Implement your changes**
5. **Run the test suite**: `make test-cov`
6. **Submit a pull request**

---

## Code Style

### Python

- Python 3.12+ only
- Use type hints for all public functions
- Follow PEP 8 (enforced by flake8)
- Max line length: 120 characters

### Naming

- `snake_case` for functions and variables
- `PascalCase` for classes
- `UPPER_CASE` for constants
- Descriptive names over abbreviations

### Comments

- Comments explain **why**, not **what**
- Docstrings for all public functions
- Keep docstrings concise

### Example

```python
# Author: Your Name

def compute_belief_decay(
    confidence: float,
    decay_rate: float,
    elapsed_seconds: float,
) -> float:
    """
    apply exponential decay to belief confidence.

    decay follows: c' = c * e^(-rate * time)
    """
    if confidence < 0 or confidence > 1:
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")

    import math
    return confidence * math.exp(-decay_rate * elapsed_seconds)
```

---

## Testing

### Requirements

- All new features must have tests
- All bug fixes must have regression tests
- Target: 80%+ coverage

### Running Tests

```bash
# all tests
make test

# with coverage
make test-cov

# specific module
pytest tests/test_belief_ecology.py -v

# property-based tests with stats
make property-test
```

### Hypothesis (Property-Based Testing)

We use Hypothesis for property-based testing. To reproduce a failing test:

```bash
pytest tests/test_formal_verification.py --hypothesis-seed=12345
```

Seeds are printed on failure. Store them in issue reports.

---

## Pull Request Guidelines

### Before Submitting

- [ ] Tests pass: `make test`
- [ ] Coverage maintained: `make test-cov`
- [ ] Linting passes: `make lint`
- [ ] Type checking passes: `make typecheck`
- [ ] Documentation updated if needed

### PR Description

Include:
- **What**: Brief description of changes
- **Why**: Motivation or issue reference
- **How**: High-level approach
- **Testing**: How you verified the changes

### Review Process

1. Maintainer reviews within 48 hours
2. Address feedback in new commits (don't squash during review)
3. Squash on merge

---

## Issue Guidelines

### Bug Reports

Use the bug report template. Include:
- Python version
- OS
- Minimal reproduction steps
- Expected vs actual behavior
- Error messages / stack traces

### Feature Requests

Use the feature request template. Include:
- Use case description
- Proposed solution
- Alternatives considered

### Security Issues

**Do NOT open public issues for security vulnerabilities.**

Instead:
1. Open a private security advisory on GitHub
2. Or email maintainers directly

We respond within 48 hours and aim to fix critical issues within 7 days.

---

## Architecture Notes

### Core Principles

1. **Offline-first**: No network dependencies
2. **Deterministic**: All randomness must be seeded
3. **Auditable**: All state changes logged
4. **Hard-fail**: Security features never silently downgrade

### Module Boundaries

| Module | Responsibility | Dependencies |
|--------|---------------|--------------|
| `core/` | Belief/goal/contradiction logic | numpy |
| `memory/` | Persistence, episodic replay | aiofiles |
| `security/` | Isolation, audit logging | (optional: firejail) |
| `privacy/` | DP mechanisms, budget tracking | numpy |
| `crypto/` | Signatures, commitments, proofs | cryptography |
| `verification/` | Invariants, state machine | (none) |

### Adding New Features

1. Identify which module owns the feature
2. Write invariants/properties first
3. Implement with tests
4. Update docs if public-facing

---

## Release Process

Maintainers only:

1. Update `CHANGELOG.md`
2. Bump version in relevant files
3. Run full verification: `make verify-full`
4. Tag release: `git tag v0.x.y`
5. Push tag: `git push origin v0.x.y`

---

## Getting Help

- **Issues**: For bugs and feature requests
- **Discussions**: For questions and ideas
- **Docs**: `README.md` and `docs/` folder

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

*Thank you for helping make Sentinel OS Core better!*
