# Author: Bradley R. Kinnard
# tests for soft isolation

import pytest


class TestSafeExecution:
    """test safe code execution."""

    def test_execute_safe_code(self, sandbox):
        result = sandbox.execute("x = 1 + 1")
        assert result.success is True

    def test_execute_returns_result(self, sandbox):
        result = sandbox.execute("result = 42")
        assert result.success is True
        assert result.result == 42

    def test_execute_catches_exception(self, sandbox):
        result = sandbox.execute("raise ValueError('test')")
        assert result.success is False
        assert result.error is not None


class TestFilesystemIsolation:
    """test filesystem access restrictions."""

    def test_block_filesystem_read(self, sandbox):
        result = sandbox.execute("open('/etc/passwd', 'r')")
        assert result.success is False

    def test_block_filesystem_write(self, sandbox):
        result = sandbox.execute("open('/tmp/test.txt', 'w')")
        assert result.success is False


class TestMaliciousCodeRejection:
    """test rejection of malicious code patterns."""

    def test_block_os_import(self, sandbox):
        result = sandbox.execute("import os; os.system('ls')")
        assert result.success is False
        assert "blocked" in (result.error or "").lower()

    def test_block_subprocess(self, sandbox):
        result = sandbox.execute("import subprocess; subprocess.run(['ls'])")
        assert result.success is False

    def test_block_eval(self, sandbox):
        result = sandbox.execute("eval('__import__(\"os\")')")
        assert result.success is False

    def test_block_exec(self, sandbox):
        result = sandbox.execute("exec('import os')")
        assert result.success is False

    def test_block_builtins_access(self, sandbox):
        result = sandbox.execute("__builtins__.__import__('os')")
        assert result.success is False

    def test_block_pickle(self, sandbox):
        result = sandbox.execute("import pickle")
        assert result.success is False


class TestThreatModel:
    """test explicit threat model documentation."""

    def test_threat_model_defined(self, sandbox):
        from security.soft_isolation import ThreatModel
        assert sandbox.threat_model == ThreatModel.OPPORTUNISTIC

    def test_security_disclaimer_honest(self, sandbox):
        disclaimer = sandbox.get_security_disclaimer()
        assert "NOT DEFENDED" in disclaimer
        assert "pickle" in disclaimer.lower()

    def test_isolation_level_reported(self, sandbox):
        result = sandbox.execute("x = 1")
        assert result.isolation_level in ("python", "firejail", "blocked")


class TestTimeout:
    """test timeout enforcement."""

    def test_timeout_enforced(self, sandbox_with_short_timeout):
        result = sandbox_with_short_timeout.execute(
            "import time; time.sleep(10)"
        )
        # should fail or be blocked
        assert result.success is False or "blocked" in (result.error or "")


class TestFirejailIntegration:
    """test firejail integration when available."""

    def test_firejail_available_check(self, sandbox):
        available = sandbox.is_firejail_available()
        assert isinstance(available, bool)

    def test_firejail_execution_when_available(self, sandbox_with_firejail):
        if sandbox_with_firejail is None:
            pytest.skip("firejail not installed")

        result = sandbox_with_firejail.execute("print('hello')")
        assert result.isolation_level == "firejail"


class TestTrustBoundary:
    """test trust boundary enforcement."""

    def test_validator_rejects_invalid(self):
        from security.soft_isolation import TrustBoundary

        boundary = TrustBoundary()
        boundary.register_validator(
            "input",
            lambda x: (False, "rejected") if x == "bad" else (True, "ok")
        )

        result, ok, msg = boundary.cross_boundary("input", "bad", "external", "core")
        assert not ok
        assert msg == "rejected"

    def test_sanitizer_applied(self):
        from security.soft_isolation import TrustBoundary

        boundary = TrustBoundary()
        boundary.register_sanitizer("input", lambda x: x.strip())

        result, ok, msg = boundary.cross_boundary("input", "  data  ", "external", "core")
        assert ok
        assert result == "data"

    def test_crossings_logged(self):
        from security.soft_isolation import TrustBoundary

        boundary = TrustBoundary()
        boundary.cross_boundary("test", "data", "a", "b")

        log = boundary.get_crossing_log()
        assert len(log) == 1
        assert log[0]["boundary"] == "test"


@pytest.fixture
def sandbox():
    """fixture providing fresh sandbox instance."""
    from security.soft_isolation import SoftIsolation, IsolationConfig
    config = IsolationConfig(timeout_seconds=5.0)
    return SoftIsolation(config)


@pytest.fixture
def sandbox_with_short_timeout():
    """sandbox with very short timeout for testing."""
    from security.soft_isolation import SoftIsolation, IsolationConfig
    config = IsolationConfig(timeout_seconds=0.5)
    return SoftIsolation(config)


@pytest.fixture
def sandbox_with_firejail():
    """sandbox configured to use firejail if available."""
    from security.soft_isolation import SoftIsolation, IsolationConfig
    import shutil

    if not shutil.which("firejail"):
        return None

    config = IsolationConfig(use_firejail=True, timeout_seconds=5.0)
    return SoftIsolation(config)
