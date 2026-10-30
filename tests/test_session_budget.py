"""Session-thread token budget.

The SDK agent re-submits its whole thread on every send, so the plugin retires it once
the last call's reported input reaches the budget. These tests pin the window's source,
the budget's resolution order, the retire decision, and the two wirings a refactor would
silently drop: writing the reported tokens back, and acting on that decision.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from hermes_cursor_provider import client as client_module
from hermes_cursor_provider.client import CursorSDKClient
from test_client import fake_sdk
from test_native_offer import (
    READ,
    SEARCH,
    _assistant_event,
    _follow,
    _go,
    _install,
    _native,
    _result,
    _status,
    _stream_calls,
    _usage_event,
)

LOGGER = "hermes_cursor_provider"


@pytest.fixture(autouse=True)
def _budget_sdk(fake_sdk, monkeypatch):
    monkeypatch.delenv("HERMES_CURSOR_SESSION_TOKEN_BUDGET", raising=False)
    monkeypatch.delenv("HERMES_CURSOR_NATIVE_TOOLS", raising=False)
    monkeypatch.delenv("HERMES_CURSOR_INTERCEPT_NATIVE", raising=False)
    monkeypatch.delenv("HERMES_CURSOR_NATIVE_MAX_CALLS", raising=False)
    return fake_sdk


def _usage_of(tokens):
    return SimpleNamespace(
        input_tokens=tokens,
        output_tokens=1,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )


def _extend(first, assistant, second):
    return [
        {"role": "user", "content": first},
        {"role": "assistant", "content": assistant},
        {"role": "user", "content": second},
    ]


def test_parse_context_window_reads_k_and_bare_numbers():
    assert client_module._parse_context_window("grok-4.7@context=500k") == 500_000
    assert client_module._parse_context_window("grok-4.7@context=200k,fast=true") == 200_000
    assert client_module._parse_context_window("grok-4.7@context=120000") == 120_000
    assert client_module._parse_context_window("grok-4.7@context=1m") == 1_000_000


def test_parse_context_window_absent_or_malformed_is_none():
    assert client_module._parse_context_window("grok-4.7") is None
    assert client_module._parse_context_window("grok-4.7@fast=true") is None
    assert client_module._parse_context_window("grok-4.7@context=lots") is None
    assert client_module._parse_context_window("") is None
    assert client_module._parse_context_window(None) is None


def test_session_token_budget_follows_the_model_window(monkeypatch):
    monkeypatch.delenv("HERMES_CURSOR_SESSION_TOKEN_BUDGET", raising=False)
    assert client_module._session_token_budget("grok-4.7@context=500k") == 500_000


def test_session_token_budget_off_without_any_window(monkeypatch):
    monkeypatch.delenv("HERMES_CURSOR_SESSION_TOKEN_BUDGET", raising=False)
    assert client_module._session_token_budget("grok-4.7") is None


def test_env_budget_outranks_the_model_parameter(monkeypatch):
    """The live model string always carries context=, so an env value that lost to it
    could never act on the running configuration."""
    monkeypatch.setenv("HERMES_CURSOR_SESSION_TOKEN_BUDGET", "120000")
    assert client_module._session_token_budget("grok-4.7@context=500k") == 120_000


def test_env_budget_kill_switch(monkeypatch):
    for value in ("off", "OFF", "0", "false", "no", "none"):
        monkeypatch.setenv("HERMES_CURSOR_SESSION_TOKEN_BUDGET", value)
        assert client_module._session_token_budget("grok-4.7@context=500k") is None, value


def test_env_budget_malformed_falls_back_to_the_model(monkeypatch, caplog):
    monkeypatch.setenv("HERMES_CURSOR_SESSION_TOKEN_BUDGET", "lots")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert client_module._session_token_budget("grok-4.7@context=500k") == 500_000
    assert "HERMES_CURSOR_SESSION_TOKEN_BUDGET" in caplog.text


def test_should_retire_session_thresholds():
    retire = client_module._should_retire_session
    assert retire(499_999, 500_000) is False
    assert retire(500_000, 500_000) is True  # at the budget retires
    assert retire(500_001, 500_000) is True
    assert retire(0, 500_000) is False  # unknown never retires
    assert retire(900_000, None) is False  # policy off


def test_session_carries_the_reported_input_tokens(tmp_path):
    """The field the retire decision reads must exist and default to 'unknown'."""
    session = client_module._Session(key=("a", "b", "c", "d"), bridge=None, cwd=tmp_path)
    assert session.last_input_tokens == 0
    assert session.budget_warned is False


def test_session_records_reported_input_tokens_from_the_call():
    """The retire decision reads session.last_input_tokens, so the call must write it."""
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = "hello"
    sdk.result.usage = _usage_of(123_456)
    client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "first"}],
        _cursor_session_scope="scope-tokens",
    )
    sessions = list(client_module._SESSIONS.values())
    client.close()
    assert len(sessions) == 1
    assert sessions[0].last_input_tokens == 123_456


def test_session_retires_agent_when_reported_tokens_reach_the_budget(monkeypatch, caplog):
    """At the budget the next turn must start a fresh agent (the SDK thread is reset)."""
    monkeypatch.setenv("HERMES_CURSOR_SESSION_TOKEN_BUDGET", "1000")
    caplog.set_level(logging.INFO, logger=LOGGER)
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = "hello"
    sdk.result.usage = _usage_of(1000)  # exactly at the budget
    client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "first"}],
        _cursor_session_scope="scope-retire",
    )
    assert len(sdk.agents) == 1

    sdk.result.usage = _usage_of(10)
    client.chat.completions.create(
        model="auto",
        messages=_extend("first", "hello", "second"),
        _cursor_session_scope="scope-retire",
    )
    client.close()
    assert len(sdk.agents) == 2, "the over-budget session did not retire its agent"
    assert sdk.agents[0].closed
    assert '"content":"first"' in sdk.agents[1].messages[0]["text"]
    created = [line for line in caplog.text.splitlines() if "agent created" in line]
    assert any("reason=budget" in line for line in created)


def test_session_keeps_agent_under_the_budget(monkeypatch):
    """Below the budget the agent is reused and only the delta is sent."""
    monkeypatch.setenv("HERMES_CURSOR_SESSION_TOKEN_BUDGET", "1000")
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = "hello"
    sdk.result.usage = _usage_of(999)
    client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "first"}],
        _cursor_session_scope="scope-keep",
    )
    client.chat.completions.create(
        model="auto",
        messages=_extend("first", "hello", "second"),
        _cursor_session_scope="scope-keep",
    )
    client.close()
    assert len(sdk.agents) == 1, "an under-budget session should reuse its agent"
    assert not sdk.agents[0].closed


def test_kill_switch_keeps_one_agent_across_an_over_budget_session(monkeypatch):
    """HERMES_CURSOR_SESSION_TOKEN_BUDGET=off disables the policy on the live config."""
    monkeypatch.setenv("HERMES_CURSOR_SESSION_TOKEN_BUDGET", "off")
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = "hello"
    sdk.result.usage = _usage_of(10_000_000)  # far past any window
    client.chat.completions.create(
        model="grok-4.7@context=500k",
        messages=[{"role": "user", "content": "first"}],
        _cursor_session_scope="scope-off",
    )
    client.chat.completions.create(
        model="grok-4.7@context=500k",
        messages=_extend("first", "hello", "second"),
        _cursor_session_scope="scope-off",
    )
    client.close()
    assert len(sdk.agents) == 1, "the kill switch did not disable the policy"


def test_session_with_no_reported_tokens_never_retires(monkeypatch):
    """0 means 'unknown' (the SDK reported nothing), not 'over budget'.

    A fresh session whose first call reports no usage must keep its agent: retiring on
    unknown would reset the thread every turn and re-send the whole transcript each time.
    """
    monkeypatch.setenv("HERMES_CURSOR_SESSION_TOKEN_BUDGET", "1000")
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = "hello"
    sdk.result.usage = SimpleNamespace(input_tokens=0, output_tokens=0)
    client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "first"}],
        _cursor_session_scope="scope-unknown",
    )
    client.chat.completions.create(
        model="auto",
        messages=_extend("first", "hello", "second"),
        _cursor_session_scope="scope-unknown",
    )
    client.close()
    assert len(sdk.agents) == 1, "unknown reported tokens must not retire the agent"


def test_native_turn_reuses_agent_when_billed_sum_exceeds_budget(monkeypatch):
    """Two native turns each fit the budget; their billed sum does not.

    The next request must reuse the agent. Recording the billed sum (or failing to
    record the per-turn anchor) would retire it.
    """
    monkeypatch.setenv("HERMES_CURSOR_NATIVE_TOOLS", "1")
    monkeypatch.setenv("HERMES_CURSOR_SESSION_TOKEN_BUDGET", "1000")
    client = CursorSDKClient(api_key="crsr_test")
    try:
        sdk = client._sdk_client()
        script = [
            _native("read_file", {"path": "a"}, call_id="c1"),
            _usage_event(400),
            _native("web_search", {"q": "b"}, call_id="c2"),
            _usage_event(600),
            _status("FINISHED"),
        ]
        _install(sdk, [
            (script, _result(input_tokens=1600), False),
            ([_assistant_event("ok")], _result("ok", input_tokens=10), False),
        ])
        first = _go(
            client,
            [{"role": "user", "content": "anchor-user"}],
            [READ, SEARCH],
            stream=True,
            scope="scope-anchor",
        )
        calls = _stream_calls(first)
        assert [call.function.name for call in calls] == ["read_file", "web_search"]
        sessions = list(client_module._SESSIONS.values())
        assert len(sessions) == 1
        # Peak per-turn anchor (600), not the wait() billed sum (1600).
        assert sessions[0].last_input_tokens == 600
        follow = _follow(
            "anchor-user",
            first,
            {"role": "tool", "tool_call_id": calls[0].id, "content": "a"},
            {"role": "tool", "tool_call_id": calls[1].id, "content": "b"},
            {"role": "user", "content": "next"},
        )
        _go(client, follow, [READ, SEARCH], stream=True, scope="scope-anchor")
        assert len(sdk.agents) == 1, "a native turn under the per-turn anchor must reuse its agent"
        assert not sdk.agents[0].closed
    finally:
        client.close()


def test_budget_retire_clears_inherited_input_tokens(monkeypatch):
    """A retired agent must not keep the count that forced the retire.

    The fresh run fails before the success path can overwrite the field, so the only
    way the count is 0 is the retire itself clearing it.
    """
    monkeypatch.setenv("HERMES_CURSOR_SESSION_TOKEN_BUDGET", "1000")
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = "hello"
    sdk.result.usage = _usage_of(1000)
    client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "first"}],
        _cursor_session_scope="scope-reset",
    )
    sessions = list(client_module._SESSIONS.values())
    assert sessions[0].last_input_tokens == 1000

    from test_client import FakeAgent

    def create_agent(options):
        sdk.options.append(options)
        agent = FakeAgent(sdk)
        sdk.agents.append(agent)

        def send(_message):
            raise RuntimeError("fresh run failed")

        agent.send = send
        return agent

    sdk.create_agent = create_agent
    with pytest.raises(RuntimeError, match="fresh run failed"):
        client.chat.completions.create(
            model="auto",
            messages=_extend("first", "hello", "second"),
            _cursor_session_scope="scope-reset",
        )
    client.close()
    assert sessions[0].last_input_tokens == 0
    assert sessions[0].agent is None
