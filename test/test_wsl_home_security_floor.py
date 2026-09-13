"""WSL Windows-home floor: the Windows profile is a second fenced home.

A WSL2 gateway sees the operator's Windows credential stores through DrvFs
(``/mnt/c/Users/<you>/.aws``, ...) at spellings no Linux-home anchor names.
These tests pin that the whole floor -- the read gate AND the sandbox mask --
re-anchors under that root, following the ``KIROCREW_OS_HOME`` precedent in
``test_pod_home_remap_security_floor.py`` (an alternate whole home re-anchors
EVERY entry, keyed through ``_ResolvedRoots`` so the cache cannot serve one
root's targets under another root's key).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from kiro_crew.security import paths as sec_paths

WIN_SSH_KEY = os.path.join(".ssh", "id_rsa")
PI_TOKEN_LEAF = ".pi/agent/auth.json"

#: The real interop query, saved before the fixture below neuters it.
#: Interop tests restore this so the fake ``subprocess`` they install is
#: actually reached (a fixture-neutered query would return ``None`` without
#: ever spawning, and every interop test would pass vacuously).
_REAL_QUERY = sec_paths._query_interop_windows_home


@pytest.fixture()
def wsl_case(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Pin the Linux home, force WSL detection on, neutralize interop.

    ``USERPROFILE`` is pinned alongside ``HOME`` because Windows
    ``Path.home()`` reads that one and never ``HOME`` (same reason as the
    pod-floor helper). Interop is neutralized so every non-interop test is
    hermetic without Win32 binaries; interop tests restore ``_REAL_QUERY``
    explicitly. Both caches are dropped so the pinned roots actually reach
    the builders.
    """
    home = tmp_path / "linux-home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr("kiro_crew.sandbox.is_wsl", lambda: True)
    monkeypatch.setattr(sec_paths, "_query_interop_windows_home", lambda: None)
    sec_paths._reset_wsl_home_cache()
    sec_paths._home_targets_cache.clear()
    yield tmp_path
    sec_paths._reset_wsl_home_cache()
    sec_paths._home_targets_cache.clear()


def _make_win_profile(base: Path, name: str = "winhome") -> Path:
    win = base / name
    (win / ".aws").mkdir(parents=True, exist_ok=True)
    (win / ".ssh").mkdir(parents=True, exist_ok=True)
    (win / ".pi" / "agent").mkdir(parents=True, exist_ok=True)
    (win / "project").mkdir(parents=True, exist_ok=True)
    return win


def _use_real_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore the real interop query (undoing the fixture's neuter) so a
    test-installed fake ``subprocess`` is reached. Resets the memo so the
    neutered answer cannot leak into the test."""
    monkeypatch.setattr(sec_paths, "_query_interop_windows_home", _REAL_QUERY)
    sec_paths._reset_wsl_home_cache()


class TestWslHomeReadGate:
    def test_override_env_anchors_credential_dirs(
        self, monkeypatch: pytest.MonkeyPatch, wsl_case: Path
    ) -> None:
        """Every floor entry -- OS stores and harness token leaves alike --
        is refused under the Windows profile root."""
        win = _make_win_profile(wsl_case)
        monkeypatch.setenv(sec_paths.WSL_WINDOWS_HOME_ENV, str(win))
        assert sec_paths.is_sensitive_path(str(win / ".aws" / "credentials")) is True
        assert sec_paths.is_sensitive_path(str(win / WIN_SSH_KEY)) is True
        assert sec_paths.is_sensitive_path(str(win / PI_TOKEN_LEAF)) is True

    def test_ordinary_windows_home_files_stay_readable(
        self, monkeypatch: pytest.MonkeyPatch, wsl_case: Path
    ) -> None:
        """The anchor is a second home, not a second world: non-credential
        paths under it are unaffected."""
        win = _make_win_profile(wsl_case)
        monkeypatch.setenv(sec_paths.WSL_WINDOWS_HOME_ENV, str(win))
        assert sec_paths.is_sensitive_path(str(win / "project" / "main.py")) is False

    def test_env_ignored_off_wsl(
        self, monkeypatch: pytest.MonkeyPatch, wsl_case: Path
    ) -> None:
        """Native hosts keep byte-identical behavior however the override is
        set: the variable names WSL, and off WSL it must be inert."""
        win = _make_win_profile(wsl_case)
        monkeypatch.setenv(sec_paths.WSL_WINDOWS_HOME_ENV, str(win))
        monkeypatch.setattr("kiro_crew.sandbox.is_wsl", lambda: False)
        sec_paths._home_targets_cache.clear()
        assert sec_paths.is_sensitive_path(str(win / ".aws" / "credentials")) is False
        assert sec_paths._resolve_wsl_home() is None


class TestWslHomeMask:
    def test_mask_covers_windows_home_leaves(
        self, monkeypatch: pytest.MonkeyPatch, wsl_case: Path
    ) -> None:
        """The OS mask denies the same leaves under the Windows profile that
        it denies under the Linux home."""
        win = _make_win_profile(wsl_case)
        monkeypatch.setenv(sec_paths.WSL_WINDOWS_HOME_ENV, str(win))
        win_real = os.path.realpath(str(win))
        targets = sec_paths.sandbox_credential_targets()
        assert os.path.join(win_real, ".aws") in targets
        assert os.path.join(win_real, ".ssh") in targets
        assert os.path.join(win_real, ".pi", "agent", "auth.json") in targets

    def test_mask_exclusion_applies_under_both_roots(
        self, monkeypatch: pytest.MonkeyPatch, wsl_case: Path
    ) -> None:
        """An adapter's own leaf is spared from the mask everywhere it is
        anchored -- Linux home AND Windows profile -- while the rest of the
        Windows-home cover stays up."""
        win = _make_win_profile(wsl_case)
        monkeypatch.setenv(sec_paths.WSL_WINDOWS_HOME_ENV, str(win))
        win_real = os.path.realpath(str(win))
        targets = sec_paths.sandbox_credential_targets(exclude_leaves=(PI_TOKEN_LEAF,))
        assert os.path.join(win_real, ".pi", "agent", "auth.json") not in targets
        assert os.path.join(win_real, ".aws") in targets

    def test_no_env_no_interop_adds_nothing(self, wsl_case: Path) -> None:
        """WSL without a determinable profile is exactly the base mask: the
        interop arm answering ``None`` must not move a single target."""
        assert sec_paths._resolve_wsl_home() is None
        base = sec_paths.sandbox_credential_targets()
        assert all("/winhome/" not in t and "\\winhome\\" not in t for t in base)

    def test_roots_carry_wsl_home(
        self, monkeypatch: pytest.MonkeyPatch, wsl_case: Path
    ) -> None:
        """The resolved root is a cache-key field: two different profiles
        build two different target sets (serving one root's set under the
        other's key would be the fail-OPEN shape the key exists to prevent)."""
        win_a = _make_win_profile(wsl_case, "winhome-a")
        win_b = _make_win_profile(wsl_case, "winhome-b")
        monkeypatch.setenv(sec_paths.WSL_WINDOWS_HOME_ENV, str(win_a))
        roots_a = sec_paths._resolve_root_anchors(str(Path.home()))
        assert roots_a.wsl_home == os.path.realpath(str(win_a))
        set_a = set(sec_paths.sandbox_credential_targets())
        sec_paths._home_targets_cache.clear()
        monkeypatch.setenv(sec_paths.WSL_WINDOWS_HOME_ENV, str(win_b))
        roots_b = sec_paths._resolve_root_anchors(str(Path.home()))
        assert roots_b.wsl_home == os.path.realpath(str(win_b))
        set_b = set(sec_paths.sandbox_credential_targets())
        aws_a = os.path.join(os.path.realpath(str(win_a)), ".aws")
        aws_b = os.path.join(os.path.realpath(str(win_b)), ".aws")
        assert aws_a in set_a and aws_a not in set_b
        assert aws_b in set_b and aws_b not in set_a

    def test_rejects_filesystem_root(
        self, monkeypatch: pytest.MonkeyPatch, wsl_case: Path
    ) -> None:
        """``/`` -- which lexes to the bare drive root on Windows, whose
        ``isabs`` calls it drive-relative -- would fence every workspace
        into unreadability: ``None`` on every platform while the base
        anchors hold."""
        monkeypatch.setenv(sec_paths.WSL_WINDOWS_HOME_ENV, "/")
        assert sec_paths._resolve_wsl_home() is None

    @pytest.mark.skipif(
        os.name == "nt",
        reason="a Windows spelling is native there, not foreign",
    )
    def test_rejects_foreign_spellings(
        self, monkeypatch: pytest.MonkeyPatch, wsl_case: Path
    ) -> None:
        """A Windows spelling is meaningless to a POSIX anchor (a backslash
        is an ordinary filename character there): ``None``. Skipped on
        Windows, where the same spelling is native and absolute."""
        monkeypatch.setenv(sec_paths.WSL_WINDOWS_HOME_ENV, "C:\\Users\\someone")
        assert sec_paths._resolve_wsl_home() is None


class _FakeModule:
    """A stand-in for the ``subprocess`` name inside ``paths``.

    Patched into the ``paths`` namespace only, so the global interpreter is
    untouched and no real process ever spawns.
    """

    def __init__(self, run: object) -> None:
        self.run = run
        self.TimeoutExpired = subprocess.TimeoutExpired
        self.SubprocessError = subprocess.SubprocessError
        self.CompletedProcess = subprocess.CompletedProcess


class _FakeRun:
    """Scripted ``subprocess.run`` for the interop query."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.profile_stdout = "C:\\Users\\t\r\n"
        self.profile_rc = 0
        self.wslpath_stdout = "/mnt/c/Users/t\n"
        self.wslpath_rc = 0
        self.raise_on: str | None = None

    def __call__(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        if self.raise_on is not None and argv[0] == self.raise_on:
            raise subprocess.TimeoutExpired(argv, timeout=kwargs.get("timeout"))
        if argv[0] == "cmd.exe":
            return subprocess.CompletedProcess(argv, self.profile_rc, self.profile_stdout, "")
        assert argv[0] == "wslpath", argv
        return subprocess.CompletedProcess(argv, self.wslpath_rc, self.wslpath_stdout, "")


class TestInteropDerivation:
    def test_query_maps_profile_to_drvfs(
        self, monkeypatch: pytest.MonkeyPatch, wsl_case: Path
    ) -> None:
        """The two-hop derivation (``cmd.exe`` names it, ``wslpath``
        converts it) lands on the DrvFs spelling."""
        _use_real_query(monkeypatch)
        fake = _FakeRun()
        monkeypatch.setattr(sec_paths, "subprocess", _FakeModule(fake))
        assert sec_paths._query_interop_windows_home() == "/mnt/c/Users/t"
        assert fake.calls[0][0] == "cmd.exe"
        assert fake.calls[1][0] == "wslpath"

    def test_unexpanded_profile_and_wslpath_failure_answer_none(
        self, monkeypatch: pytest.MonkeyPatch, wsl_case: Path
    ) -> None:
        """A literal ``%USERPROFILE%`` never reaches ``wslpath`` (its output
        on junk would be a guess), and a failing ``wslpath`` is ``None``."""
        _use_real_query(monkeypatch)
        fake = _FakeRun()
        fake.profile_stdout = "%USERPROFILE%\r\n"
        monkeypatch.setattr(sec_paths, "subprocess", _FakeModule(fake))
        assert sec_paths._query_interop_windows_home() is None
        assert [c[0] for c in fake.calls] == ["cmd.exe"]
        fake2 = _FakeRun()
        fake2.wslpath_rc = 1
        monkeypatch.setattr(sec_paths, "subprocess", _FakeModule(fake2))
        assert sec_paths._query_interop_windows_home() is None

    def test_timeout_retries_instead_of_memoizing(
        self, monkeypatch: pytest.MonkeyPatch, wsl_case: Path
    ) -> None:
        """One slow ``cmd.exe`` costs this build's cover, not the process's:
        the timeout is not memoized, so the next rebuild derives the root."""
        _use_real_query(monkeypatch)
        fake = _FakeRun()
        fake.raise_on = "cmd.exe"
        monkeypatch.setattr(sec_paths, "subprocess", _FakeModule(fake))
        assert sec_paths._memoized_interop_windows_home() is None
        fake.raise_on = None
        assert sec_paths._memoized_interop_windows_home() == "/mnt/c/Users/t"
        assert len(fake.calls) == 3

    def test_success_and_hard_failure_memoize(
        self, monkeypatch: pytest.MonkeyPatch, wsl_case: Path
    ) -> None:
        """Successes and permanent failures (no interop binaries) run the
        query once per process; only timeouts re-probe."""
        _use_real_query(monkeypatch)
        fake = _FakeRun()
        monkeypatch.setattr(sec_paths, "subprocess", _FakeModule(fake))
        assert sec_paths._memoized_interop_windows_home() == "/mnt/c/Users/t"
        assert sec_paths._memoized_interop_windows_home() == "/mnt/c/Users/t"
        assert len(fake.calls) == 2

        sec_paths._reset_wsl_home_cache()

        class _NoInterop:
            def __call__(self, argv: list[str], **kwargs: object) -> object:
                raise FileNotFoundError("no interop here")

        monkeypatch.setattr(sec_paths, "subprocess", _FakeModule(_NoInterop()))
        assert sec_paths._memoized_interop_windows_home() is None
        assert sec_paths._memoized_interop_windows_home() is None
