"""The pi backend offers, and is held to, the model list pi-acp advertises.

Same three seams as the codex picker (``test_codex_advertised_models.py``),
one of which pi never needed. pi-acp advertises its models only as a
``configOptions`` ``model`` select on ``session/new`` -- ``provider/id``
pairs -- and that list is the ONLY vocabulary
``session/set_config_option("model")`` accepts. Before the adapter change:

* the capture harvested a single-current select, so the picker could only
  offer the running model back;
* ``GET /api/models`` fell through to kiro-cli's ``--list-models`` catalog,
  so the picker offered kiro ids pi has never heard of;
* a pick from that catalog reached the wire at startup; pi-acp answers the
  worded ``Invalid value for config option model: ...`` the push helper
  already reads as a value rejection (unlike codex's bare ``-32602``), so no
  new rejection shape was needed -- only the offered list had to become true.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kiro_crew import model_registry
from kiro_crew.acp.client import (
    DEFAULT_MODEL,
    AcpClient,
    AcpError,
    AcpModelUnavailable,
)
from kiro_crew.acp.types import (
    ACP_BACKEND_CLAUDE,
    ACP_BACKEND_CODEX,
    ACP_BACKEND_KIRO,
    ACP_BACKEND_PI,
)
from kiro_crew.agent_sdk import backends as sdk_backends
from kiro_crew.agent_sdk.capabilities import capabilities_for
from kiro_crew.dashboard.handlers import agents

#: What pi-acp puts on the wire at ``session/new`` once it advertises the
#: catalog: values are the ``provider/id`` pairs ``set_config_option`` accepts
#: back, names are the bare ids, descriptions name the provider.
PI_SESSION_NEW = {
    "sessionId": "ses_pi_9",
    "configOptions": [
        {
            "id": "mode",
            "type": "select",
            "currentValue": "read-only",
            "options": [{"value": "read-only", "name": "Read only"}],
        },
        {
            "id": "model",
            "type": "select",
            "currentValue": "opencode-go/kimi-k2.6",
            "options": [
                {
                    "value": "opencode-go/kimi-k2.6",
                    "name": "kimi-k2.6",
                    "description": "opencode-go",
                },
                {
                    "value": "opencode-go/kimi-k3",
                    "name": "kimi-k3",
                    "description": "opencode-go",
                },
                {
                    "value": "anthropic/claude-opus-5",
                    "name": "claude-opus-5",
                    "description": "anthropic",
                },
            ],
        },
        {
            "id": "effort",
            "type": "select",
            "currentValue": "medium",
            "options": [{"value": "medium", "name": "Medium"}],
        },
    ],
}


@pytest.fixture(autouse=True)
def _cold_advertised_cache(monkeypatch):
    """Every test here starts from an empty cross-session cache.

    ``model_registry._ADVERTISED_MODELS`` is a module global other tests on the
    same xdist worker feed; these tests assert on which BUCKET gets fed, so a
    warm one would pass or fail on a neighbour's leftovers.
    """
    monkeypatch.setattr(model_registry, "_ADVERTISED_MODELS", {})
    monkeypatch.setattr(model_registry, "persist_advertised_models", lambda: None)


def _pi_client(tmp_path, model: str = "") -> AcpClient:
    client = AcpClient(work_dir=tmp_path, acp_backend=ACP_BACKEND_PI)
    client._session_id = "ses_pi_9"
    client._model = model
    return client


# ── Seam 1: the capability sets ──


def test_pi_is_an_advertised_model_selection_member() -> None:
    assert ACP_BACKEND_PI in sdk_backends.ACP_BACKENDS_ADVERTISED_MODEL_SELECTION
    assert capabilities_for(ACP_BACKEND_PI).resolves_model_from_advertised_list is True


def test_pi_has_its_own_registry_namespace() -> None:
    """The namespace is also the advertised-cache bucket: pi's ``provider/id``
    pairs must not land in the ``acp`` bucket a kiro-family harness would read
    back."""
    assert sdk_backends.model_registry_namespace(ACP_BACKEND_PI) == "pi"
    assert capabilities_for(ACP_BACKEND_PI).model_id_namespace == "pi"
    # The kiro-family answer is untouched (harness-parity H13).
    assert sdk_backends.model_registry_namespace(ACP_BACKEND_KIRO) == "acp"


def test_pi_membership_does_not_drag_in_the_settings_seed() -> None:
    """The two opt-ins are independent: pi writes no settings.local.json."""
    assert ACP_BACKEND_PI not in sdk_backends.ACP_BACKENDS_SEED_LOCAL_SETTINGS


# ── Seam 2: the capture ──


def test_pi_capture_harvests_the_config_options_model_select(tmp_path) -> None:
    client = _pi_client(tmp_path)

    client._capture_available_models(PI_SESSION_NEW)

    assert [m["modelId"] for m in client.available_models()] == [
        "opencode-go/kimi-k2.6",
        "opencode-go/kimi-k3",
        "anthropic/claude-opus-5",
    ]
    assert client._resolved_model_id == "opencode-go/kimi-k2.6"


def test_pi_capture_feeds_the_pi_cache_bucket_only(tmp_path) -> None:
    client = _pi_client(tmp_path)

    client._capture_available_models(PI_SESSION_NEW)

    assert model_registry.advertised_models("pi") == [
        "opencode-go/kimi-k2.6",
        "opencode-go/kimi-k3",
        "anthropic/claude-opus-5",
    ]
    assert model_registry.advertised_models("acp") == []
    assert client._advertised_models_changed is True


def test_kiro_capture_still_ignores_a_config_options_select(tmp_path) -> None:
    """The kiro path did not move (harness-parity H13)."""
    client = AcpClient(work_dir=tmp_path, acp_backend=ACP_BACKEND_KIRO)

    client._capture_available_models(PI_SESSION_NEW)

    assert client.available_models() == []
    assert model_registry.advertised_models("acp") == []


# ── Seam 3: the worded value rejection ──


def _refusing_with(message: str):
    """A ``set_config_option`` double that refuses every value the pi way."""
    applied: list[tuple[str, str]] = []

    async def _set_config_option(config_id: str, value: str) -> None:
        applied.append((config_id, value))
        raise AcpError(message)

    return _set_config_option, applied


class TestStartupModelPushOnPi:
    @pytest.mark.asyncio
    async def test_worded_rejection_withholds_and_keeps_the_session(self, tmp_path) -> None:
        """A stale kiro id pushed to pi must not kill init: pi-acp's worded
        rejection is read as a value verdict, the session stays on the backend
        default, and the pin is recorded as inheriting."""
        client = _pi_client(tmp_path, model="gpt-5.6-sol")
        client._resolved_model_id = "opencode-go/kimi-k2.6"
        set_option, applied = _refusing_with(
            "Invalid value for config option model: gpt-5.6-sol (unknown pi model gpt-5.6-sol)"
        )
        client.set_config_option = set_option  # type: ignore[method-assign]

        await client._apply_startup_model()

        assert applied == [("model", "gpt-5.6-sol")]
        assert client._model == DEFAULT_MODEL
        assert client._resolved_model_id == "opencode-go/kimi-k2.6"

    @pytest.mark.asyncio
    async def test_worded_rejection_on_an_explicit_pick_is_typed(self, tmp_path) -> None:
        """``set_model`` is the user's own pick: refusal is ``AcpModelUnavailable``,
        never a generic error the caller would answer with a session reset."""
        client = _pi_client(tmp_path, model="opencode-go/kimi-k2.6")
        client._capture_available_models(PI_SESSION_NEW)
        set_option, _applied = _refusing_with(
            "Invalid value for config option model: nope/not-a-model (unknown pi model nope/not-a-model)"
        )
        client.set_config_option = set_option  # type: ignore[method-assign]

        with pytest.raises(AcpModelUnavailable) as info:
            await client.set_model("nope/not-a-model")

        assert info.value.advertised == [
            "opencode-go/kimi-k2.6",
            "opencode-go/kimi-k3",
            "anthropic/claude-opus-5",
        ]
        assert client._model == "opencode-go/kimi-k2.6"

    @pytest.mark.asyncio
    async def test_an_accepted_value_is_recorded(self, tmp_path) -> None:
        client = _pi_client(tmp_path, model="opencode-go/kimi-k3")
        applied: list[tuple[str, str]] = []

        async def _accept(config_id: str, value: str) -> None:
            applied.append((config_id, value))

        client.set_config_option = _accept  # type: ignore[method-assign]

        await client._apply_startup_model()

        assert applied == [("model", "opencode-go/kimi-k3")]
        assert client._model == "opencode-go/kimi-k3"


# ── The picker: GET /api/models on pi ──


def _request(*providers) -> MagicMock:
    state = SimpleNamespace(sessions=SimpleNamespace(active_providers=lambda: list(providers)))
    request = MagicMock()
    request.app = {"state": state}
    return request


def _pi_provider(tmp_path) -> MagicMock:
    """A provider double wrapping a client that captured a real pi session/new."""
    client = _pi_client(tmp_path)
    client._capture_available_models(PI_SESSION_NEW)
    provider = MagicMock()
    # ``_advertised_cc_models`` selects on the capability record, and a
    # MagicMock's attributes are all truthy -- so hand it the real record.
    provider.capabilities = capabilities_for(ACP_BACKEND_PI)
    provider.available_models = MagicMock(return_value=client.available_models())
    return provider


def _names(rows: list[dict]) -> list[str]:
    return [r["model_name"] for r in rows]


def test_pi_picker_lists_the_live_session_advertised_ids(tmp_path) -> None:
    rows = agents._pi_models(_request(_pi_provider(tmp_path)))

    assert _names(rows) == [
        "auto",
        "opencode-go/kimi-k2.6",
        "opencode-go/kimi-k3",
        "anthropic/claude-opus-5",
    ]
    assert rows[1]["display_name"] == "kimi-k2.6"
    assert rows[1]["description"] == "opencode-go"
    assert all(isinstance(r["context_window"], int) and r["context_window"] > 0 for r in rows)


def test_pi_picker_reads_the_cross_session_cache_when_no_session_is_live() -> None:
    """A dashboard restarted after a pi session still offers the real list."""
    model_registry.refresh_advertised_models("pi", ["opencode-go/kimi-k3"])

    rows = agents._pi_models(_request())

    assert _names(rows) == ["auto", "opencode-go/kimi-k3"]


def test_pi_picker_never_reads_the_kiro_bucket() -> None:
    model_registry.refresh_advertised_models("acp", ["claude-opus-5", "gpt-5.6-sol"])

    rows = agents._pi_models(_request())

    assert _names(rows) == ["auto"]


def test_pi_picker_cold_offers_auto_alone() -> None:
    """No session yet and nothing cached: ``auto`` (inherit pi's default) only.
    The frontend refetches on the first session spawn."""
    assert _names(agents._pi_models(_request())) == ["auto"]


def test_pi_picker_resurrects_the_configured_default_only_when_nothing_is_known() -> None:
    cold = agents._pi_models(_request(), configured_default="opencode-go/kimi-k3")
    assert _names(cold) == ["auto", "opencode-go/kimi-k3"]
    assert cold[1]["description"] == "Configured default"

    model_registry.refresh_advertised_models("pi", ["opencode-go/kimi-k2.6"])
    known = agents._pi_models(_request(), configured_default="gpt-5.6-sol")
    # The stale kiro pin is exactly the row that kills the session: not offered.
    assert _names(known) == ["auto", "opencode-go/kimi-k2.6"]


def test_pi_picker_does_not_duplicate_auto_or_a_configured_advertised_id() -> None:
    model_registry.refresh_advertised_models("pi", ["auto", "opencode-go/kimi-k2.6", "opencode-go/kimi-k2.6"])

    rows = agents._pi_models(_request(), configured_default="opencode-go/kimi-k2.6")

    assert _names(rows) == ["auto", "opencode-go/kimi-k2.6"]


def test_pi_picker_ignores_a_provider_without_the_capability(tmp_path) -> None:
    """A kiro session's list is a different vocabulary and must not be offered."""
    kiro = MagicMock()
    kiro.capabilities = capabilities_for(ACP_BACKEND_KIRO)
    kiro.available_models = MagicMock(return_value=[{"modelId": "claude-opus-5"}])

    assert _names(agents._pi_models(_request(kiro))) == ["auto"]


def _foreign_session(backend: str, ids: list[str]) -> MagicMock:
    """A live session of another harness holding the same capability."""
    other = MagicMock()
    other.capabilities = capabilities_for(backend)
    other.available_models = MagicMock(
        return_value=[{"modelId": i, "name": i, "description": ""} for i in ids]
    )
    return other


def test_pi_picker_ignores_a_claude_session_that_holds_the_same_capability() -> None:
    """claude also resolves its model from its advertised list; only
    ``model_id_namespace`` says whose ids these are."""
    claude = _foreign_session(ACP_BACKEND_CLAUDE, ["claude-opus-5"])

    assert _names(agents._pi_models(_request(claude))) == ["auto"]


def test_pi_picker_ignores_a_codex_session_that_holds_the_same_capability() -> None:
    """codex likewise: offering its ids on pi would put back rows pi refuses."""
    codex = _foreign_session(ACP_BACKEND_CODEX, ["gpt-5.4"])

    assert _names(agents._pi_models(_request(codex))) == ["auto"]


def test_pi_picker_prefers_the_newest_pi_session(tmp_path) -> None:
    """Two live pi sessions: the later one's snapshot wins."""
    older = MagicMock()
    older.capabilities = capabilities_for(ACP_BACKEND_PI)
    older.available_models = MagicMock(
        return_value=[{"modelId": "opencode-go/kimi-k2.6", "name": "kimi-k2.6", "description": ""}]
    )

    rows = agents._pi_models(_request(older, _pi_provider(tmp_path)))

    assert _names(rows) == [
        "auto",
        "opencode-go/kimi-k2.6",
        "opencode-go/kimi-k3",
        "anthropic/claude-opus-5",
    ]


@pytest.mark.asyncio
async def test_api_models_routes_the_pi_backend_to_the_advertised_list(monkeypatch, tmp_path):
    """The handler branch: pi never reaches the kiro-cli ``--list-models`` spawn."""
    import json

    monkeypatch.setattr(
        agents.KiroCrewConfig,
        "load",
        staticmethod(
            lambda: SimpleNamespace(agent=SimpleNamespace(acp_backend=ACP_BACKEND_PI, model=""))
        ),
    )

    async def _never_spawn(*_a, **_k):  # pragma: no cover - the assertion is that it is unreached
        raise AssertionError("pi must not spawn kiro-cli --list-models")

    monkeypatch.setattr(agents, "reject_if_kiro_unverified", _never_spawn)

    resp = await agents.api_models(_request(_pi_provider(tmp_path)))

    assert resp.status == 200
    assert _names(json.loads(resp.body)) == [
        "auto",
        "opencode-go/kimi-k2.6",
        "opencode-go/kimi-k3",
        "anthropic/claude-opus-5",
    ]
