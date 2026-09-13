"""``kirocrew doctor`` reports pi as the DEFAULT agent backend.

pi is the floor every deployment keeps, so its install row must exist
unconditionally: a missing default with no row would leave the operator whose
sessions all fail with nothing naming the absent component. These tests call
``_doctor_pi_backend`` rather than ``_doctor()``, and stub the probe, for the
same reason ``test_doctor_claude_backend.py`` does -- the full doctor reaches
the host.
"""

import contextlib
import io

import pytest

from kiro_crew import cli_doctor
from kiro_crew.agent_sdk import INSTALLED, MISSING, UNKNOWN, BackendInstallState


def _state(installed: str, **over) -> BackendInstallState:
    return BackendInstallState(
        backend="pi",
        policy_id="pi",
        installed=installed,
        **over,
    )


def _report(monkeypatch, state) -> str:
    """Run the reporting block with the probe stubbed, capturing stdout.

    ``state=None`` stands for a probe that raised: the function catches it and must
    still print a line rather than staying silent.
    """

    def _probe(_backend):
        if state is None:
            raise RuntimeError("probe blew up")
        return state

    monkeypatch.setattr("kiro_crew.agent_sdk.probe_backend", _probe)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with contextlib.suppress(SystemExit):
            cli_doctor._doctor_pi_backend()
    return buf.getvalue()


def test_an_install_names_the_default_backend(monkeypatch):
    out = _report(monkeypatch, _state(INSTALLED))
    assert "pi-acp:" in out
    assert "the default agent backend" in out


def test_reports_the_harness_when_it_is_absent(monkeypatch):
    """The line exists unconditionally, so a missing default is discoverable."""
    out = _report(
        monkeypatch,
        _state(MISSING, missing_components=("pi-acp",)),
    )
    assert "pi-acp:" in out
    assert "pi-acp not found (the default agent backend)" in out


def test_a_failed_check_is_not_reported_as_absent(monkeypatch):
    """``unknown`` means the probe could not answer, which is not evidence of absence --
    the same three-valued contract the dashboard honours."""
    out = _report(monkeypatch, _state(UNKNOWN))
    assert "not found" not in out
    assert "could not check" in out


def test_a_raising_probe_still_prints_a_line(monkeypatch):
    """Doctor degrades to "could not check" rather than dropping the row silently."""
    out = _report(monkeypatch, None)
    assert "could not check" in out


def test_the_reporting_does_not_run_the_whole_doctor():
    """Guard the reason this file calls the helper: ``_doctor`` reaches the host.

    If someone folds the block back inline, this test's own import of the helper
    fails -- which is the signal, not a style preference.
    """
    assert callable(cli_doctor._doctor_pi_backend)


@pytest.mark.parametrize("verdict", [INSTALLED, MISSING, UNKNOWN])
def test_no_verdict_is_a_hard_failure(monkeypatch, verdict):
    """Doctor reports install state, so no verdict may raise -- even for the default."""
    out = _report(monkeypatch, _state(verdict, missing_components=("pi-acp",)))
    assert "pi-acp:" in out


def test_an_install_is_never_called_selectable(monkeypatch):
    """Doctor reads the INSTALL probe, so it may not claim selectability.

    Whether the deployment may select the backend is a separate answer that
    ``apply_selectable_denials`` can say no to, and doctor never consults it.
    """
    out = _report(monkeypatch, _state(INSTALLED))
    assert "installed" in out
    assert "selectable" not in out
