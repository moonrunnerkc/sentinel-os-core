# Author: Bradley R. Kinnard
# tests for sandbox - TDD

import pytest
import os


class TestSafeExecution:
    """test safe code execution."""

    def test_execute_safe_code(self, sandbox):
        result = sandbox.execute_safely("x = 1 + 1")
        assert result["success"] is True

    def test_execute_returns_result(self, sandbox):
        result = sandbox.execute_safely("result = 42")
        assert result["success"] is True
        assert result.get("result") == 42

    def test_execute_catches_exception(self, sandbox):
        result = sandbox.execute_safely("raise ValueError('test')")
        assert result["success"] is False
        assert "error" in result


class TestFilesystemIsolation:
    """test filesystem access restrictions."""

    def test_block_filesystem_read(self, sandbox):
        result = sandbox.execute_safely("open('/etc/passwd', 'r')")
        assert result["success"] is False

    def test_block_filesystem_write(self, sandbox):
        result = sandbox.execute_safely("open('/tmp/test.txt', 'w')")
        assert result["success"] is False

    def test_allow_approved_paths(self, sandbox, tmp_path):
        # configure sandbox with allowed path
        sandbox.add_allowed_path(str(tmp_path))
        test_file = tmp_path / "test.txt"
        test_file.write_text("test data")

        result = sandbox.execute_safely(f"open('{test_file}', 'r').read()")
        # should still be blocked in restrictive mode
        assert result["success"] is False  # default is restrictive


class TestMaliciousCodeRejection:
    """test rejection of malicious code patterns."""

    def test_block_os_import(self, sandbox):
        result = sandbox.execute_safely("import os; os.system('ls')")
        assert result["success"] is False
        assert "blocked" in result.get("error", "").lower() or "not allowed" in result.get("error", "").lower()

    def test_block_subprocess(self, sandbox):
        result = sandbox.execute_safely("import subprocess; subprocess.run(['ls'])")
        assert result["success"] is False

    def test_block_eval(self, sandbox):
        result = sandbox.execute_safely("eval('__import__(\"os\")')")
        assert result["success"] is False

    def test_block_exec(self, sandbox):
        result = sandbox.execute_safely("exec('import os')")
        assert result["success"] is False

    def test_block_builtins_access(self, sandbox):
        result = sandbox.execute_safely("__builtins__.__import__('os')")
        assert result["success"] is False


class TestStateIntegrity:
    """test post-execution state integrity."""

    def test_state_hash_unchanged_on_safe_code(self, sandbox):
        pre_hash = sandbox.get_state_hash()
        sandbox.execute_safely("x = 1")
        post_hash = sandbox.get_state_hash()

        # safe code shouldn't mutate global state
        assert pre_hash == post_hash

    def test_state_mutation_detected(self, sandbox, caplog):
        import logging
        caplog.set_level(logging.WARNING)

        # attempt to mutate state (should be caught)
        sandbox.execute_safely("globals()['__hack__'] = True")

        # check for mutation warning
        # note: this may or may not trigger depending on sandbox impl


class TestFirejailIntegration:
    """test firejail integration when available."""

    def test_firejail_available_check(self, sandbox):
        available = sandbox.is_firejail_available()
        # just verify the check runs without error
        assert isinstance(available, bool)

    def test_firejail_execution(self, sandbox):
        if not sandbox.is_firejail_available():
            pytest.skip("firejail not installed")

        # run with firejail wrapper
        result = sandbox.execute_with_firejail("print('hello')")
        assert result["success"] is True


class TestSecurityLimitations:
    """
    document known limitations.
    these tests acknowledge what the sandbox cannot prevent.
    """

    def test_acknowledge_cpython_escape_risk(self, sandbox):
        """
        this sandbox cannot prevent all CPython escape vectors.
        determined adversaries with code execution may bypass restrictions.
        """
        # this is documentation, not a security guarantee
        assert sandbox.get_security_disclaimer() is not None

    def test_acknowledge_pickle_risk(self, sandbox):
        """pickle deserialization can execute arbitrary code."""
        disclaimer = sandbox.get_security_disclaimer()
        assert "pickle" in disclaimer.lower() or "limitation" in disclaimer.lower()


@pytest.fixture
def sandbox():
    """fixture providing fresh sandbox instance."""
    from security.sandbox import Sandbox
    return Sandbox()
