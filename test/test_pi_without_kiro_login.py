"""Using the pi backend must not require Kiro sign-in — scoped, not global.

pi brings its own provider keys (``~/.pi/agent``, BYO ``ANTHROPIC_API_KEY``, …
— pi-acp's "Auth model"); there is no Kiro subscription involved, so any
device-code sign-in gate, Kiro-identity requirement, or host-auth demand on the
pi backend's path is friction with no security function. The seam is
``ACP_BACKENDS_BYO_AUTH`` (:mod:`kiro_crew.agent_sdk.backends`): other
self-authenticating backends join later, one deliberate membership each —
never a blanket removal. kiro-cli, KAS membership, and the host-auth callback
keep every existing requirement.

Every bypassed gate below is proved in BOTH directions, in the membership-test
style of the pi parity slices:

* (a) the pi path works with no Kiro identity present — a BYO-auth member that
  is still selectable skips the Kiro readiness gates;
* (b) the kiro/KAS paths still demand it — every current backend, every
  unresolvable value, and a member denied back to kiro stay gated.

Fail closed throughout: with the seam empty (its main state) nothing bypasses,
and a member that stops resolving to itself (governance denial, unregistered
build) degrades to kiro and stays gated with it.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kiro_crew.acp_backends import (
    ACP_BACKEND_CLAUDE,
    ACP_BACKEND_CODEX,
    ACP_BACKEND_KAS,
    ACP_BACKEND_KIRO,
    ACP_BACKEND_OPENCODE,
    ACP_BACKENDS_ACP_RUNTIME,
    ACP_BACKENDS_HOST_AUTH_CALLBACK,
    bypasses_kiro_signin_gate,
)
from kiro_crew.agent_sdk import backends as sdk_backends
from kiro_crew.config.loader import KiroCrewConfig
from kiro_crew.dashboard import kiro_readiness

#: The pi spelling, used as a literal on purpose: ``ACP_BACKEND_PI`` is defined
#: on the ``feat/pi-acp-selectable`` side (with its ``ACP_BACKENDS_KNOWN``
#: registration, per H8). Spelling it here pins the wiring for the id the seam
#: is built for, so the join is a membership edit rather than a re-verification.
_PI = "pi"


@pytest.fixture
def _byo_pi(monkeypatch):
    """Stand in for the parity side's membership edit: pi joins the seam."""
    monkeypatch.setattr(sdk_backends, "ACP_BACKENDS_BYO_AUTH", frozenset({_PI}))
    return _PI


@pytest.fixture
def _selectable_pi(monkeypatch):
    """Stand in for the parity side's registration: pi stays selectable."""
    monkeypatch.setattr(sdk_backends, "_selectable", set(sdk_backends._selectable) | {_PI})


def _fake_config(*, default: str = "", member: str = "kas"):
    return SimpleNamespace(agent=SimpleNamespace(acp_backend=default, member_acp_backend=member))


@pytest.fixture
def _config(monkeypatch):
    """Point the bypass helper's config load at a fake agent section."""
    state: dict[str, object] = {"cfg": _fake_config()}

    def _set(*, default: str = "", member: str = "kas") -> None:
        state["cfg"] = _fake_config(default=default, member=member)

    monkeypatch.setattr(KiroCrewConfig, "load", classmethod(lambda cls: state["cfg"]))
    return _set


# ── the predicate: every current backend stays gated ────────────────────────


@pytest.mark.parametrize(
    "backend",
    [
        ACP_BACKEND_KIRO,
        ACP_BACKEND_KAS,
        ACP_BACKEND_CLAUDE,
        ACP_BACKEND_CODEX,
        ACP_BACKEND_OPENCODE,
        "something-unknown",
        "",
        None,
        123,
        {"backend": "pi"},
    ],
)
def test_current_and_unresolvable_backends_stay_gated(backend: object) -> None:
    """(b) kiro, KAS, every BYO-but-not-yet-joined backend, and any misshapen
    configured value keep every Kiro-sign-in requirement."""
    assert bypasses_kiro_signin_gate(backend) is False


def test_pi_stays_gated_while_unselectable(_byo_pi: str) -> None:
    """Fail closed: membership alone bypasses nothing.

    On main pi is neither registered nor selectable, so even as a seam member
    it degrades to kiro at session start -- and that turn needs Kiro sign-in.
    The bypass fires only once the parity side's registration lands.
    """
    assert bypasses_kiro_signin_gate(_byo_pi) is False


def test_pi_bypasses_once_member_and_selectable(_byo_pi: str, _selectable_pi: None) -> None:
    """(a) the pi path with no Kiro identity present skips the Kiro gates."""
    assert bypasses_kiro_signin_gate(_byo_pi) is True


def test_pi_stays_gated_when_denied_back_to_kiro(_byo_pi: str) -> None:
    """Fail closed: a governance-denied member degrades to kiro, so it gates.

    ``apply_selectable_denials`` recomputes the selectable set as
    ``baseline - denied``; a denied pi resolves to kiro, and the turn that
    starts there needs Kiro sign-in. Membership must not outvote selectability.
    """
    sdk_backends.apply_selectable_denials({_PI})
    try:
        assert bypasses_kiro_signin_gate(_byo_pi) is False
    finally:
        sdk_backends.apply_selectable_denials(set())


# ── pi must never touch the Kiro-identity surfaces ──────────────────────────


def test_pi_never_takes_the_kiro_runtime_path() -> None:
    """pi runs per-session (AcpClient), never on the kiro-family AcpRuntime.

    The ``saw_not_logged_in`` → ``AcpAuthRequired`` translation lives on the
    runtime path only; non-membership is what keeps a pi session's own auth
    vocabulary from ever being read as a Kiro sign-in demand.
    """
    assert _PI not in ACP_BACKENDS_ACP_RUNTIME


def test_pi_is_never_handed_crew_credential() -> None:
    """Only KAS may answer ``_kiro/auth/getAccessToken`` from Crew's vault.

    pi authenticates from its own files like opencode does; handing it Crew's
    credential would cross the exact boundary the host-auth callback exists to
    hold. A pi session that reaches a Kiro-credentialed surface must error,
    never silently inherit -- and non-membership answers such a request
    method-not-found, never with a token.
    """
    assert _PI not in ACP_BACKENDS_HOST_AUTH_CALLBACK
    assert ACP_BACKENDS_HOST_AUTH_CALLBACK == frozenset({ACP_BACKEND_KAS})


# ── the per-session helper: member arm mirrors the provider factory ─────────


@pytest.mark.asyncio
async def test_helper_gates_everything_with_no_membership(_config) -> None:
    """(b) with the seam empty, no session on any backend bypasses."""
    _config(default="", member="kas")
    assert await kiro_readiness.session_bypasses_kiro_readiness("chat-1") is False
    assert await kiro_readiness.session_bypasses_kiro_readiness("member-x") is False
    assert await kiro_readiness.session_bypasses_kiro_readiness(None) is False


@pytest.mark.asyncio
async def test_helper_bypasses_default_pi_session(
    _config, _byo_pi: str, _selectable_pi: None
) -> None:
    """(a) an ordinary pi session skips the Kiro gates with no Kiro identity."""
    _config(default=_byo_pi, member="kas")
    assert await kiro_readiness.session_bypasses_kiro_readiness("chat-1") is True
    assert await kiro_readiness.session_bypasses_kiro_readiness(None) is True


@pytest.mark.asyncio
async def test_helper_bypasses_member_pi_session(
    _config, _byo_pi: str, _selectable_pi: None
) -> None:
    """(a) a member DM on the pi member backend takes the member arm, like the
    provider factory's own ``select_provider_backend`` does."""
    _config(default="", member=_byo_pi)
    assert await kiro_readiness.session_bypasses_kiro_readiness("member-x") is True
    assert await kiro_readiness.session_bypasses_kiro_readiness("dashboard:member-x") is True
    # ...while the default arm on the same config stays gated.
    assert await kiro_readiness.session_bypasses_kiro_readiness("chat-1") is False


@pytest.mark.asyncio
async def test_helper_stays_gated_for_denied_member_backend(_config, _byo_pi: str) -> None:
    """Fail closed: a member backend denied back to kiro keeps the gate."""
    _config(default="", member=_byo_pi)
    assert await kiro_readiness.session_bypasses_kiro_readiness("member-x") is False


@pytest.mark.asyncio
async def test_helper_fails_closed_on_unreadable_config(monkeypatch) -> None:
    """Fail closed: if identity state is ambiguous, keep demanding sign-in."""

    def _boom(cls):
        raise OSError("config home wedged")

    monkeypatch.setattr(KiroCrewConfig, "load", classmethod(_boom))
    assert await kiro_readiness.session_bypasses_kiro_readiness("chat-1") is False


# ── the gated endpoints, driven for real while signed out ───────────────────


def _signed_out_service():
    """A filesystem-free not-ready prerequisite service (the gate's latch)."""
    from kiro_crew.kiro_prerequisite import KiroPrerequisiteService

    class _SignedOut(KiroPrerequisiteService):
        async def session_ready(self) -> bool:
            return False

        async def verified_ready(self, *, max_age_secs: float) -> bool:
            del max_age_secs
            return False

    return object.__new__(_SignedOut)


def _slot_request(service, slot: str) -> MagicMock:
    request = MagicMock()
    request.app = {
        "kiro_prerequisite_service": service,
        "state": SimpleNamespace(_slots={}),
    }
    request.match_info = {"slot": slot}
    request.get.return_value = ""
    return request


@pytest.mark.asyncio
async def test_regenerate_still_gated_for_kiro_while_signed_out(_config) -> None:
    """(b) a kiro session's destructive rerun keeps its 503 while signed out."""
    from kiro_crew.dashboard.chat_regenerate import api_chat_slot_regenerate

    _config(default="", member="kas")
    resp = await api_chat_slot_regenerate(_slot_request(_signed_out_service(), "chat-1"))
    assert resp.status == 503


@pytest.mark.asyncio
async def test_regenerate_bypassed_for_pi_while_signed_out(
    _config, _byo_pi: str, _selectable_pi: None
) -> None:
    """(a) a pi session's rerun is not held behind Kiro sign-in.

    Past the gate the missing slot 404s -- which is exactly the proof: the
    gated path would have answered 503 first.
    """
    from kiro_crew.dashboard.chat_regenerate import api_chat_slot_regenerate

    _config(default=_byo_pi, member="kas")
    resp = await api_chat_slot_regenerate(_slot_request(_signed_out_service(), "chat-1"))
    assert resp.status == 404


@pytest.mark.asyncio
async def test_edit_resend_bypassed_for_pi_while_signed_out(
    _config, _byo_pi: str, _selectable_pi: None
) -> None:
    """(a) same bypass for edit-resend; (b) kiro stays gated (parametrized)."""
    from kiro_crew.dashboard.chat_regenerate import api_chat_slot_edit_resend

    _config(default=_byo_pi, member="kas")
    resp = await api_chat_slot_edit_resend(_slot_request(_signed_out_service(), "chat-1"))
    assert resp.status == 404


@pytest.mark.asyncio
async def test_edit_resend_still_gated_for_kiro_while_signed_out(_config) -> None:
    """(b) kiro edit-resend keeps its 503 while signed out."""
    from kiro_crew.dashboard.chat_regenerate import api_chat_slot_edit_resend

    _config(default="", member="kas")
    resp = await api_chat_slot_edit_resend(_slot_request(_signed_out_service(), "chat-1"))
    assert resp.status == 503


@pytest.mark.asyncio
async def test_rewind_bypassed_for_pi_while_signed_out(
    _config, _byo_pi: str, _selectable_pi: None
) -> None:
    """(a) same bypass for rewind: past the gate, the missing slot 404s."""
    from kiro_crew.dashboard.chat_rewind import api_chat_slot_rewind

    _config(default=_byo_pi, member="kas")
    resp = await api_chat_slot_rewind(_slot_request(_signed_out_service(), "chat-1"))
    assert resp.status == 404


@pytest.mark.asyncio
async def test_rewind_still_gated_for_kiro_while_signed_out(_config) -> None:
    """(b) kiro rewind keeps its 503 while signed out."""
    from kiro_crew.dashboard.chat_rewind import api_chat_slot_rewind

    _config(default="", member="kas")
    resp = await api_chat_slot_rewind(_slot_request(_signed_out_service(), "chat-1"))
    assert resp.status == 503


def _compat_request(service, body: dict) -> MagicMock:
    request = MagicMock()
    request.app = {
        "kiro_prerequisite_service": service,
        "state": SimpleNamespace(),
    }
    request.get.return_value = ""
    request.json = AsyncMock(return_value=body)
    return request


@pytest.mark.asyncio
async def test_completions_still_gated_for_kiro_while_signed_out(_config) -> None:
    """(b) the compat endpoint keeps its OpenAI-shaped 503 while signed out."""
    import json as _json

    from kiro_crew.dashboard.openai_compat import api_completions

    _config(default="", member="kas")
    resp = await api_completions(
        _compat_request(_signed_out_service(), {"model": "", "messages": []})
    )
    assert resp.status == 503
    payload = _json.loads(resp.text)
    assert payload["error"]["code"] == "kiro_prerequisite_required"


@pytest.mark.asyncio
async def test_completions_bypassed_for_pi_while_signed_out(
    _config, _byo_pi: str, _selectable_pi: None
) -> None:
    """(a) a pi compat call is not held behind Kiro sign-in.

    Past the gate the empty model 400s on validation -- which is the proof:
    the gated path would have answered 503 first, and the peeked body still
    reaches the real parse below, so malformed input keeps its status.
    """
    from kiro_crew.dashboard.openai_compat import api_completions

    _config(default=_byo_pi, member="kas")
    resp = await api_completions(
        _compat_request(_signed_out_service(), {"model": "", "messages": []})
    )
    assert resp.status == 400


@pytest.mark.asyncio
async def test_completions_malformed_body_keeps_its_status_when_gated(
    _config,
) -> None:
    """Peeking never upgrades a malformed body: unparsable input on a kiro
    gateway still answers the gate's 503 (and would 400 past it)."""
    from kiro_crew.dashboard.openai_compat import api_completions

    _config(default="", member="kas")
    request = _compat_request(_signed_out_service(), {})
    request.json = AsyncMock(side_effect=ValueError("no json"))
    resp = await api_completions(request)
    assert resp.status == 503
