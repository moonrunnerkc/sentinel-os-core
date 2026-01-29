# Author: Bradley R. Kinnard
# pytest configuration and fixtures

"""
Test Configuration

Hypothesis Settings:
- Default seed: controlled via pytest-randomly or explicit seed
- Reproducibility: run with --hypothesis-seed=<seed> to reproduce
- Database: .hypothesis/ stores examples for shrinking

To reproduce a failing test:
  pytest tests/test_formal_verification.py --hypothesis-seed=12345

To see Hypothesis statistics:
  pytest tests/ --hypothesis-show-statistics
"""

import os
import pytest
from hypothesis import settings, Phase

# configure hypothesis defaults
settings.register_profile(
    "ci",
    max_examples=100,
    deadline=None,  # disable deadline in CI (slower machines)
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
    print_blob=True,  # print blob for reproduction
)

settings.register_profile(
    "dev",
    max_examples=20,
    deadline=2000,  # 2s deadline for dev
)

settings.register_profile(
    "extensive",
    max_examples=500,
    deadline=None,
)

# load profile from environment or default to dev
profile = os.environ.get("HYPOTHESIS_PROFILE", "dev")
settings.load_profile(profile)


@pytest.fixture(scope="session")
def hypothesis_seed():
    """
    document how to get reproducible hypothesis runs.

    usage:
        pytest --hypothesis-seed=12345

    the seed will be printed on failure for reproduction.
    """
    import hypothesis
    return hypothesis._settings.default.database


@pytest.fixture
def seeded_rng():
    """provide a seeded random generator for deterministic tests."""
    import numpy as np
    rng = np.random.default_rng(42)
    return rng


@pytest.fixture
def temp_data_dir(tmp_path):
    """provide a temporary data directory for tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "beliefs").mkdir()
    (data_dir / "episodes").mkdir()
    (data_dir / "logs").mkdir()
    return data_dir


def pytest_configure(config):
    """add custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "chaos: marks tests as chaos/fault-injection tests"
    )
    config.addinivalue_line(
        "markers", "requires_firejail: requires firejail to be installed"
    )
    config.addinivalue_line(
        "markers", "requires_docker: requires docker to be running"
    )
    config.addinivalue_line(
        "markers", "requires_liboqs: requires liboqs-python to be installed"
    )


def pytest_collection_modifyitems(config, items):
    """warn about skipped tests at collection time."""
    skip_reasons = []
    for item in items:
        for marker in item.iter_markers("skip"):
            reason = marker.kwargs.get("reason", "no reason given")
            skip_reasons.append(f"  - {item.name}: {reason}")

    if skip_reasons:
        print("\n" + "=" * 60)
        print("⚠️  SKIPPED TESTS WARNING")
        print("=" * 60)
        print("The following tests will be skipped:")
        for reason in skip_reasons[:10]:  # limit output
            print(reason)
        if len(skip_reasons) > 10:
            print(f"  ... and {len(skip_reasons) - 10} more")
        print("=" * 60 + "\n")
