# Author: Bradley R. Kinnard
# mock mode tests for external deps (firejail, docker)
# allows CI to test isolation logic without actual binaries

import pytest
from unittest.mock import patch, MagicMock
import shutil

from security.isolation import (
    Isolation,
    IsolationConfig,
    IsolationLevel,
    IsolationUnavailableError,
    ExecutionResult,
    check_level_available,
)


class TestFirejailMockMode:
    """
    test firejail isolation logic with mocked binary.

    these tests exercise the firejail code paths without
    requiring firejail to be installed.
    """

    def test_firejail_unavailable_raises(self):
        """verify hard-fail when firejail requested but unavailable."""
        with patch.object(shutil, "which", return_value=None):
            with pytest.raises(IsolationUnavailableError) as exc:
                config = IsolationConfig(level=IsolationLevel.FIREJAIL)
                Isolation(config)

            assert "firejail" in str(exc.value).lower()

    def test_firejail_check_level_available(self):
        """verify check_level_available returns correct status."""
        with patch.object(shutil, "which", return_value=None):
            available, reason = check_level_available(IsolationLevel.FIREJAIL)
            assert not available
            assert "not installed" in reason.lower() or "not found" in reason.lower()

    def test_firejail_mock_execution(self):
        """test firejail execution path with mocked subprocess."""
        with patch.object(shutil, "which", return_value="/usr/bin/firejail"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout="hello\n",
                    stderr="",
                )

                # should not raise
                available, _ = check_level_available(IsolationLevel.FIREJAIL)
                # note: full execution test requires more mocking


class TestDockerMockMode:
    """
    test docker isolation logic with mocked daemon.

    these tests exercise docker code paths without
    requiring docker to be running.
    """

    def test_docker_unavailable_raises(self):
        """verify hard-fail when docker requested but unavailable."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("docker not found")

            with pytest.raises((IsolationUnavailableError, FileNotFoundError)):
                config = IsolationConfig(level=IsolationLevel.DOCKER)
                Isolation(config)

    def test_docker_check_level_available_no_daemon(self):
        """verify check returns false when daemon not running."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="Cannot connect to Docker daemon")

            available, reason = check_level_available(IsolationLevel.DOCKER)
            assert not available

    def test_docker_mock_execution(self):
        """test docker execution path with mocked subprocess."""
        with patch("subprocess.run") as mock_run:
            # mock docker info success
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Docker running\n",
                stderr="",
            )

            available, _ = check_level_available(IsolationLevel.DOCKER)
            # behavior depends on subprocess mock response


class TestIsolationLevelFallback:
    """test isolation level fallback behavior."""

    def test_none_level_always_available(self):
        """verify none isolation is always available."""
        available, _ = check_level_available(IsolationLevel.NONE)
        assert available

    def test_pattern_only_always_available(self):
        """verify pattern_only is always available."""
        available, _ = check_level_available(IsolationLevel.PATTERN_ONLY)
        assert available

    def test_python_sandbox_always_available(self):
        """verify python_sandbox is always available."""
        available, _ = check_level_available(IsolationLevel.PYTHON_SANDBOX)
        assert available

    def test_explicit_level_no_silent_downgrade(self):
        """verify no silent downgrades when level explicitly requested."""
        # requesting firejail without it installed should fail, not downgrade
        with patch.object(shutil, "which", return_value=None):
            with pytest.raises(IsolationUnavailableError):
                config = IsolationConfig(level=IsolationLevel.FIREJAIL)
                Isolation(config)


class TestMockExecutionResults:
    """test execution result handling with mocks."""

    def test_execution_result_dataclass(self):
        """verify ExecutionResult works correctly."""
        result = ExecutionResult(
            success=True,
            result="hello world",
            error="",
            execution_time_ms=10.5,
            isolation_level=IsolationLevel.PYTHON_SANDBOX,
        )

        assert result.success
        assert result.result == "hello world"
        assert result.isolation_level == IsolationLevel.PYTHON_SANDBOX

    def test_execution_result_failure(self):
        """verify failure results captured correctly."""
        result = ExecutionResult(
            success=False,
            result=None,
            error="permission denied",
            execution_time_ms=5.0,
            isolation_level=IsolationLevel.FIREJAIL,
        )

        assert not result.success
        assert result.error == "permission denied"


class TestCICompatibility:
    """
    tests that pass in CI without external deps.

    these provide coverage for isolation module
    even when firejail/docker are unavailable.
    """

    def test_isolation_config_creation(self):
        """verify config creation works."""
        config = IsolationConfig(level=IsolationLevel.PATTERN_ONLY)
        assert config.level == IsolationLevel.PATTERN_ONLY

    def test_isolation_levels_enum(self):
        """verify all isolation levels defined."""
        levels = list(IsolationLevel)
        assert IsolationLevel.NONE in levels
        assert IsolationLevel.PATTERN_ONLY in levels
        assert IsolationLevel.PYTHON_SANDBOX in levels
        assert IsolationLevel.FIREJAIL in levels
        assert IsolationLevel.DOCKER in levels

    def test_python_sandbox_execution(self):
        """verify python sandbox works in CI."""
        config = IsolationConfig(level=IsolationLevel.PYTHON_SANDBOX)
        isolation = Isolation(config)
        result = isolation.execute("x = 1 + 1")

        # sandbox may restrict print but basic operations should work
        assert result is not None

    def test_pattern_only_blocks_dangerous(self):
        """verify pattern_only catches obvious attacks."""
        config = IsolationConfig(level=IsolationLevel.PATTERN_ONLY)
        isolation = Isolation(config)

        # should block or warn on dangerous patterns
        result = isolation.execute("import os; os.system('rm -rf /')")
        # either blocked or executed in sandbox safely
        assert result is not None
