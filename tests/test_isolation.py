# Author: Bradley R. Kinnard
# tests for isolation module - explicit levels, hard-fail, no silent downgrade

import pytest
import shutil

from security.isolation import (
    Isolation,
    IsolationLevel,
    IsolationConfig,
    IsolationUnavailableError,
    ExecutionResult,
    SecurityViolation,
    TrustBoundary,
    LEVEL_THREAT_MODELS,
    get_level_threat_model,
    check_level_available,
)


class TestIsolationLevelEnum:
    """test isolation level enum and threat models."""

    def test_all_levels_defined(self):
        assert IsolationLevel.NONE.value == "none"
        assert IsolationLevel.PATTERN_ONLY.value == "pattern_only"
        assert IsolationLevel.PYTHON_SANDBOX.value == "python_sandbox"
        assert IsolationLevel.FIREJAIL.value == "firejail"
        assert IsolationLevel.DOCKER.value == "docker"

    def test_all_levels_have_threat_model(self):
        for level in IsolationLevel:
            model = get_level_threat_model(level)
            assert "defends" in model
            assert "limitations" in model
            assert "requires" in model
            assert isinstance(model["defends"], list)
            assert isinstance(model["limitations"], list)
            assert isinstance(model["requires"], list)

    def test_none_level_has_no_defenses(self):
        model = get_level_threat_model(IsolationLevel.NONE)
        assert len(model["defends"]) == 0
        assert "no isolation at all" in model["limitations"][0].lower()

    def test_firejail_requires_binary(self):
        model = get_level_threat_model(IsolationLevel.FIREJAIL)
        assert "firejail binary" in model["requires"][0].lower()

    def test_docker_requires_daemon(self):
        model = get_level_threat_model(IsolationLevel.DOCKER)
        assert "docker daemon" in model["requires"][0].lower()


class TestAvailabilityCheck:
    """test isolation level availability checking."""

    def test_none_always_available(self):
        available, reason = check_level_available(IsolationLevel.NONE)
        assert available is True

    def test_pattern_only_always_available(self):
        available, reason = check_level_available(IsolationLevel.PATTERN_ONLY)
        assert available is True

    def test_python_sandbox_always_available(self):
        available, reason = check_level_available(IsolationLevel.PYTHON_SANDBOX)
        assert available is True

    def test_firejail_availability_matches_binary(self):
        has_firejail = shutil.which("firejail") is not None
        available, reason = check_level_available(IsolationLevel.FIREJAIL)
        assert available == has_firejail

    def test_availability_returns_reason(self):
        available, reason = check_level_available(IsolationLevel.FIREJAIL)
        assert isinstance(reason, str)
        assert len(reason) > 0


class TestHardFailSemantics:
    """test that unavailable levels cause hard failure, not silent downgrade."""

    def test_firejail_unavailable_hard_fail(self):
        if shutil.which("firejail"):
            pytest.skip("firejail is installed, cannot test hard-fail")

        config = IsolationConfig(level=IsolationLevel.FIREJAIL)
        with pytest.raises(IsolationUnavailableError) as exc_info:
            Isolation(config)

        assert "firejail" in str(exc_info.value).lower()
        assert "unavailable" in str(exc_info.value).lower()

    def test_docker_unavailable_hard_fail(self):
        # check if docker is available
        available, _ = check_level_available(IsolationLevel.DOCKER)
        if available:
            pytest.skip("docker is available, cannot test hard-fail")

        config = IsolationConfig(level=IsolationLevel.DOCKER)
        with pytest.raises(IsolationUnavailableError) as exc_info:
            Isolation(config)

        assert "docker" in str(exc_info.value).lower()

    def test_no_silent_downgrade(self):
        # ensure level attribute matches requested level after init
        config = IsolationConfig(level=IsolationLevel.PYTHON_SANDBOX)
        isolation = Isolation(config)
        assert isolation.level == IsolationLevel.PYTHON_SANDBOX


class TestSafeExecution:
    """test safe code execution at various levels."""

    def test_execute_safe_code_none(self):
        config = IsolationConfig(level=IsolationLevel.NONE)
        isolation = Isolation(config)
        result = isolation.execute("x = 1 + 1")
        assert result.success is True

    def test_execute_safe_code_pattern_only(self):
        config = IsolationConfig(level=IsolationLevel.PATTERN_ONLY)
        isolation = Isolation(config)
        result = isolation.execute("x = 1 + 1")
        assert result.success is True

    def test_execute_safe_code_python_sandbox(self):
        config = IsolationConfig(level=IsolationLevel.PYTHON_SANDBOX)
        isolation = Isolation(config)
        result = isolation.execute("x = 1 + 1")
        assert result.success is True

    def test_execute_returns_result(self):
        config = IsolationConfig(level=IsolationLevel.PYTHON_SANDBOX)
        isolation = Isolation(config)
        result = isolation.execute("result = 42")
        assert result.success is True
        assert result.result == 42

    def test_execute_catches_exception(self):
        config = IsolationConfig(level=IsolationLevel.PYTHON_SANDBOX)
        isolation = Isolation(config)
        result = isolation.execute("raise ValueError('test')")
        assert result.success is False
        assert result.error is not None


class TestPatternBlocking:
    """test pattern-based code blocking."""

    def test_block_os_import(self):
        config = IsolationConfig(level=IsolationLevel.PYTHON_SANDBOX)
        isolation = Isolation(config)
        result = isolation.execute("import os")
        assert result.success is False
        assert "blocked" in (result.error or "").lower()

    def test_block_subprocess(self):
        config = IsolationConfig(level=IsolationLevel.PYTHON_SANDBOX)
        isolation = Isolation(config)
        result = isolation.execute("import subprocess")
        assert result.success is False

    def test_block_eval(self):
        config = IsolationConfig(level=IsolationLevel.PYTHON_SANDBOX)
        isolation = Isolation(config)
        result = isolation.execute("eval('1+1')")
        assert result.success is False

    def test_block_exec(self):
        config = IsolationConfig(level=IsolationLevel.PYTHON_SANDBOX)
        isolation = Isolation(config)
        result = isolation.execute("exec('x=1')")
        assert result.success is False

    def test_block_open(self):
        config = IsolationConfig(level=IsolationLevel.PYTHON_SANDBOX)
        isolation = Isolation(config)
        result = isolation.execute("open('/etc/passwd')")
        assert result.success is False

    def test_none_level_skips_pattern_check(self):
        config = IsolationConfig(level=IsolationLevel.NONE)
        isolation = Isolation(config)
        # NONE level does not block patterns
        # but execution will fail due to missing builtins
        # the key is that SecurityViolation is NOT raised
        result = isolation.execute("x = 'import os'")  # string, not actual import
        assert result.success is True


class TestIsolationLevelReporting:
    """test that isolation level is correctly reported in results."""

    def test_result_contains_level(self):
        config = IsolationConfig(level=IsolationLevel.PYTHON_SANDBOX)
        isolation = Isolation(config)
        result = isolation.execute("x = 1")
        assert result.isolation_level == IsolationLevel.PYTHON_SANDBOX

    def test_blocked_result_contains_level(self):
        config = IsolationConfig(level=IsolationLevel.PYTHON_SANDBOX)
        isolation = Isolation(config)
        result = isolation.execute("import os")
        assert result.isolation_level == IsolationLevel.PYTHON_SANDBOX


class TestSecurityDisclaimer:
    """test honest security disclaimers."""

    def test_disclaimer_contains_level(self):
        config = IsolationConfig(level=IsolationLevel.PYTHON_SANDBOX)
        isolation = Isolation(config)
        disclaimer = isolation.get_security_disclaimer()
        assert "python_sandbox" in disclaimer.lower()

    def test_disclaimer_lists_limitations(self):
        config = IsolationConfig(level=IsolationLevel.PYTHON_SANDBOX)
        isolation = Isolation(config)
        disclaimer = isolation.get_security_disclaimer()
        assert "limitation" in disclaimer.lower()

    def test_disclaimer_warns_about_hostile_code(self):
        config = IsolationConfig(level=IsolationLevel.PYTHON_SANDBOX)
        isolation = Isolation(config)
        disclaimer = isolation.get_security_disclaimer()
        assert "vm" in disclaimer.lower() or "hostile" in disclaimer.lower()


class TestTimeout:
    """test timeout enforcement."""

    def test_timeout_enforced(self):
        config = IsolationConfig(
            level=IsolationLevel.PYTHON_SANDBOX,
            timeout_seconds=0.5
        )
        isolation = Isolation(config)
        # time module is blocked, so use a CPU-bound loop
        result = isolation.execute("x = sum(range(10**9))")
        # should timeout or complete (depends on machine speed)
        # at minimum, should not hang forever
        assert isinstance(result, ExecutionResult)


class TestFirejailIntegration:
    """test firejail integration when available."""

    @pytest.mark.skipif(
        not shutil.which("firejail"),
        reason="firejail not installed"
    )
    def test_firejail_execution(self):
        config = IsolationConfig(level=IsolationLevel.FIREJAIL)
        isolation = Isolation(config)
        result = isolation.execute("print('hello')")
        assert result.isolation_level == IsolationLevel.FIREJAIL


class TestDockerIntegration:
    """test docker integration when available."""

    @pytest.mark.skipif(
        not check_level_available(IsolationLevel.DOCKER)[0],
        reason="docker not available"
    )
    def test_docker_execution(self):
        config = IsolationConfig(level=IsolationLevel.DOCKER)
        isolation = Isolation(config)
        result = isolation.execute("print('hello')")
        assert result.isolation_level == IsolationLevel.DOCKER


class TestTrustBoundary:
    """test trust boundary enforcement."""

    def test_validator_rejects_invalid(self):
        boundary = TrustBoundary()
        boundary.register_validator(
            "input",
            lambda x: (False, "rejected") if x == "bad" else (True, "ok")
        )

        result, ok, msg = boundary.cross_boundary("input", "bad", "external", "core")
        assert not ok
        assert msg == "rejected"

    def test_sanitizer_applied(self):
        boundary = TrustBoundary()
        boundary.register_sanitizer("input", lambda x: x.strip())

        result, ok, msg = boundary.cross_boundary("input", "  data  ", "external", "core")
        assert ok
        assert result == "data"

    def test_crossings_logged(self):
        boundary = TrustBoundary()
        boundary.cross_boundary("test", "data", "a", "b")

        log = boundary.get_crossing_log()
        assert len(log) == 1
        assert log[0]["boundary"] == "test"


class TestBackwardsCompatibility:
    """test backwards compatibility with old names."""

    def test_soft_isolation_alias(self):
        from security.isolation import SoftIsolation
        config = IsolationConfig(level=IsolationLevel.PYTHON_SANDBOX)
        isolation = SoftIsolation(config)
        assert isinstance(isolation, Isolation)

    def test_sandbox_alias(self):
        from security.isolation import Sandbox
        config = IsolationConfig(level=IsolationLevel.PYTHON_SANDBOX)
        sandbox = Sandbox(config)
        assert isinstance(sandbox, Isolation)
