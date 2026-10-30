"""Opt-in native Hermes tools (HERMES_CURSOR_NATIVE_TOOLS).

C tests are green at bf7d61f. R tests are red there because the offer does not exist yet.
Every test clears or sets the switch family through monkeypatch (see ``_isolate_switches``).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from cursor_sdk.types import CustomToolContext

from hermes_cursor_provider import client as client_module
from hermes_cursor_provider.client import CursorSDKClient, CursorSDKRunError
from test_client import (  # noqa: E402
    FakeAgent,
    FakeCursorClient,
    _assistant_event,
    _parameter,
    _stream_calls,
    _stream_text,
    fake_sdk,
)

FIXTURE = (Path(__file__).parent / "fixtures" / "prompt_bf7d61f.txt").read_text(encoding="utf-8")
INERT = (
    "Recorded for Hermes. Hermes decides whether to run it after this turn; "
    "the result, if any, arrives in the next message. Do not repeat this call."
)
LOGGER = "hermes_cursor_provider"


def _fn(name, description, prop):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {prop: {"type": "string"}}},
        },
    }


READ = _fn("read_file", "Read one file", "path")
SEARCH = _fn("web_search", "Search the web", "q")
FIXTURE_TOOLS = [READ, SEARCH]
FIXTURE_MESSAGES = [{"role": "user", "content": "inspect it"}]


@pytest.fixture(autouse=True)
def _isolate_switches(monkeypatch, fake_sdk):
    monkeypatch.delenv("HERMES_CURSOR_NATIVE_TOOLS", raising=False)
    monkeypatch.delenv("HERMES_CURSOR_INTERCEPT_NATIVE", raising=False)
    monkeypatch.delenv("HERMES_CURSOR_NATIVE_MAX_CALLS", raising=False)
    yield fake_sdk


def _switch(monkeypatch, on):
    if on:
        monkeypatch.setenv("HERMES_CURSOR_NATIVE_TOOLS", "1")
    else:
        monkeypatch.delenv("HERMES_CURSOR_NATIVE_TOOLS", raising=False)


def _usage_ns(input_tokens=4, output_tokens=2):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )


def _result(text="", *, status="FINISHED", input_tokens=4, output_tokens=2):
    return SimpleNamespace(
        result=text,
        status=status,
        status_message="",
        usage=_usage_ns(input_tokens, output_tokens),
        model=SimpleNamespace(id="auto"),
    )


class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class OfferRun:
    def __init__(self, script, result=None, *, block_wait=False):
        self.script = list(script)
        self._result = result if result is not None else _result()
        self.block_wait = block_wait
        self.cancelled = False
        self.cancel_calls = 0
        self.wait_calls = 0

    def stream(self):
        for item in self.script:
            if callable(item):
                item = item(self)
            if item is None:
                continue
            if isinstance(item, BaseException):
                raise item
            yield item

    def wait(self):
        self.wait_calls += 1
        if self.block_wait:
            import time as real_time
            real_time.sleep(0.3)
        return self._result

    def cancel(self):
        self.cancelled = True
        self.cancel_calls += 1


def _native(name, args, *, status="completed", call_id="c1"):
    return SimpleNamespace(
        type="tool_call", name=name, status=status, args=args, call_id=call_id, result=None,
    )


def _status(status, message=""):
    return SimpleNamespace(type="status", status=status, message=message)


def _thinking(text="hmm"):
    return SimpleNamespace(type="thinking", text=text)


def _usage_event(tokens):
    return SimpleNamespace(type="usage", usage=_usage_ns(tokens, 1))


def _text_block(name, args, call_id="call_t"):
    body = json.dumps({
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": args},
    })
    return f"<tool_call>{body}</tool_call>"


def _install(sdk, pending):
    """pending items are (script, result, block_wait), consumed one per send."""
    runs = []

    def create_agent(options):
        sdk.options.append(options)
        agent = FakeAgent(sdk)

        def send(message):
            agent.messages.append(message)
            sdk.messages.append(message)
            script, result, block = pending.pop(0)
            run = OfferRun(script, result, block_wait=block)
            runs.append(run)
            return run

        agent.send = send
        sdk.agents.append(agent)
        return agent

    sdk.create_agent = create_agent
    return runs


def _client():
    return CursorSDKClient(api_key="crsr_test")


def _go(client, messages, tools, *, stream=True, scope=None, tool_choice=None, model="auto"):
    kwargs = {}
    if scope:
        kwargs["_cursor_session_scope"] = scope
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    created = client.chat.completions.create(
        model=model, messages=messages, tools=tools, stream=stream, **kwargs,
    )
    return list(created) if stream else created


def _names(chunks):
    return [call.function.name for call in _stream_calls(chunks)]


def _finish(chunks):
    return [
        chunk.choices[0].finish_reason
        for chunk in chunks
        if chunk.choices and chunk.choices[0].finish_reason
    ]


def _assert_upstream_options(options):
    assert options["tools"] == []
    assert options["mcp_servers"] == {}
    assert set(options) <= {"api_key", "local", "mcp_servers", "tools", "model"}
    assert set(options["local"]) == {"cwd"}
    assert "custom_tools" not in options["local"]
    assert "setting_sources" not in options["local"]


def _follow(user, chunks, *extra):
    calls = _stream_calls(chunks)
    assistant = {
        "role": "assistant",
        "content": _stream_text(chunks),
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in calls
        ],
    }
    return [{"role": "user", "content": user}, assistant, *extra]


def _jump(clock, event):
    def produce(_run):
        clock.advance(10_000)
        return event
    return produce


# --- 1 (C) -----------------------------------------------------------------


def test_01_switch_unset_matches_upstream_prompt_and_options():
    client = _client()
    try:
        sdk = client._sdk_client()
        _go(client, FIXTURE_MESSAGES, FIXTURE_TOOLS, stream=True)
        _assert_upstream_options(sdk.options[0])
        assert sdk.messages[0]["text"] == FIXTURE
    finally:
        client.close()


# --- 2 (R) -----------------------------------------------------------------


def test_02_switch_on_offers_custom_tools_on_both_paths(monkeypatch):
    _switch(monkeypatch, True)
    client = _client()
    try:
        sdk = client._sdk_client()
        _go(client, FIXTURE_MESSAGES, FIXTURE_TOOLS, stream=True)
        _go(client, FIXTURE_MESSAGES, FIXTURE_TOOLS, stream=True, scope="offer-session")
        assert len(sdk.options) == 2
        for options in sdk.options:
            tools = options["local"]["custom_tools"]
            assert set(tools) == {"read_file", "web_search"}
            assert tools["read_file"]["description"] == "Read one file"
            assert tools["read_file"]["input_schema"]["properties"]["path"]["type"] == "string"
            assert tools["web_search"]["input_schema"] == SEARCH["function"]["parameters"]
            assert options["tools"] == ["mcp"]
            assert options["mcp_servers"] == {}
            assert "setting_sources" not in options["local"]
    finally:
        client.close()


# --- 3 (C) -----------------------------------------------------------------


def test_03_switch_on_without_tools_matches_upstream_options(monkeypatch):
    _switch(monkeypatch, True)
    client = _client()
    try:
        sdk = client._sdk_client()
        _go(client, FIXTURE_MESSAGES, None, stream=True)
        _assert_upstream_options(sdk.options[0])
    finally:
        client.close()


# --- 4 (R) -----------------------------------------------------------------


def test_04_inert_executor_returns_constant_and_writes_nothing(monkeypatch):
    _switch(monkeypatch, True)
    client = _client()
    try:
        sdk = client._sdk_client()
        _go(client, FIXTURE_MESSAGES, FIXTURE_TOOLS, stream=True, scope="inert")
        offered = sdk.options[0]["local"]["custom_tools"]
        cwd = Path(sdk.options[0]["local"]["cwd"])
        before = sorted(p.name for p in cwd.iterdir())
        for name in ("read_file", "web_search"):
            got = offered[name]["execute"]({"path": "x"}, CustomToolContext("c1"))
            assert got == INERT
            assert '"x"' not in got
        assert sorted(p.name for p in cwd.iterdir()) == before
    finally:
        client.close()


# --- 5 (R) / 6a (R) / 6b (C) ----------------------------------------------


def _script_test5(result_text):
    return [
        _native("mcp", {"toolName": "read_file", "args": {"path": "x"}}, status="running", call_id="call-a-1\nfc_b_0"),
        _native("mcp", {"toolName": "read_file", "args": {"path": "x"}}, status="completed", call_id="call-a-1\nfc_b_0"),
        _thinking(),
        _assistant_event("waiting for results"),
        _usage_event(800),
        _usage_event(1200),
        _status("FINISHED"),
    ], _result(result_text, input_tokens=2000, output_tokens=40)


def _assert_test5(chunks, run):
    calls = _stream_calls(chunks)
    assert _finish(chunks) == ["tool_calls"]
    assert len(calls) == 1
    assert calls[0].function.name == "read_file"
    assert json.loads(calls[0].function.arguments) == {"path": "x"}
    assert calls[0].id == "call-a-1"
    assert _stream_text(chunks) == ""
    assert "Still waiting" not in _stream_text(chunks)
    assert not run.cancelled
    assert chunks[-1].usage.prompt_tokens == 2000
    assert chunks[-1].usage.initial_prompt_tokens == 1200


def test_05_native_run_keeps_calls_usage_and_drops_post_text(monkeypatch):
    _switch(monkeypatch, True)
    client = _client()
    try:
        sdk = client._sdk_client()
        script, result = _script_test5("waiting for results. Still waiting.")
        runs = _install(sdk, [(script, result, False)])
        chunks = _go(client, [{"role": "user", "content": "go"}], [READ], stream=True)
        _assert_test5(chunks, runs[0])
        assert runs[0].wait_calls == 1
    finally:
        client.close()


def test_06a_switch_on_forces_interception_when_intercept_off(monkeypatch):
    _switch(monkeypatch, True)
    monkeypatch.setenv("HERMES_CURSOR_INTERCEPT_NATIVE", "0")
    client = _client()
    try:
        sdk = client._sdk_client()
        script, result = _script_test5("waiting for results. Still waiting.")
        runs = _install(sdk, [(script, result, False)])
        chunks = _go(client, [{"role": "user", "content": "go"}], [READ], stream=True)
        _assert_test5(chunks, runs[0])
    finally:
        client.close()


def test_06b_switch_off_intercept_off_ignores_native_event(monkeypatch):
    _switch(monkeypatch, False)
    monkeypatch.setenv("HERMES_CURSOR_INTERCEPT_NATIVE", "0")
    client = _client()
    try:
        sdk = client._sdk_client()
        script = [_native("read_file", {"path": "x"}, call_id="n1"), _assistant_event("plain answer")]
        _install(sdk, [(script, _result("plain answer"), False)])
        chunks = _go(client, [{"role": "user", "content": "go"}], [READ], stream=True)
        assert _stream_calls(chunks) == []
        assert "plain answer" in _stream_text(chunks)
    finally:
        client.close()


# --- 7 (R) -----------------------------------------------------------------


def test_07_real_order_keeps_distinct_calls_and_drops_repeat(monkeypatch):
    _switch(monkeypatch, True)
    client = _client()
    try:
        sdk = client._sdk_client()
        script = [
            _native("read_file", {"path": "x"}, call_id="a", status="completed"),
            _thinking(),
            _assistant_event("checking"),
            _native("web_search", {"q": "z"}, call_id="b", status="completed"),
            _native("read_file", {"path": "x"}, call_id="a-again", status="completed"),
        ]
        _install(sdk, [(script, _result(), False)])
        chunks = _go(client, [{"role": "user", "content": "go"}], [READ, SEARCH], stream=True)
        assert [(c.function.name, json.loads(c.function.arguments)) for c in _stream_calls(chunks)] == [
            ("read_file", {"path": "x"}),
            ("web_search", {"q": "z"}),
        ]
    finally:
        client.close()


# --- 8 (R) -----------------------------------------------------------------


def test_08_native_prompt_replaces_text_protocol(monkeypatch):
    _switch(monkeypatch, True)
    client = _client()
    try:
        sdk = client._sdk_client()
        _go(client, FIXTURE_MESSAGES, FIXTURE_TOOLS, stream=True)
        prompt = sdk.messages[0]["text"]
        assert "Cursor built-in tools are disabled and no MCP servers are configured." in prompt
        assert "The only tools you have are the Hermes tools listed below." in prompt
        assert "Call Hermes tools through your tool-calling channel; you may call several." in prompt
        assert "do not wait for a result or invent one" in prompt
        assert "When the work is finished, reply with the answer." in prompt
        assert "MCP servers are disabled" not in prompt
        assert "Hermes tools are NOT native tools" not in prompt
        assert "Reminder:" not in prompt
        assert "Never claim to have read, written, executed, browsed, or changed anything" in prompt
        assert prompt.rstrip().endswith("Continue from the final transcript entry.")
        assert '"name":"read_file"' in prompt and '"name":"web_search"' in prompt
    finally:
        client.close()


# --- 9a (R) / 9b (C) -------------------------------------------------------


def _miss_then_text():
    miss = _native(
        "mcp", {"toolName": "shell", "args": {"cmd": "ls"}}, status="error", call_id="miss1",
    )
    repair = _text_block("read_file", {"path": "x"}, "fixed")
    return (
        [( [miss, _status("FINISHED")], _result(), False),
         ([_assistant_event(repair)], _result(repair), False)],
    )


def test_09a_wrapped_unknown_uses_inner_name_and_native_wording(monkeypatch):
    _switch(monkeypatch, True)
    client = _client()
    try:
        sdk = client._sdk_client()
        _install(sdk, [
            ([_native("mcp", {"toolName": "shell", "args": {"cmd": "ls"}}, status="error", call_id="m1"), _status("FINISHED")], _result(), False),
            ([_assistant_event(_text_block("read_file", {"path": "x"}))], _result(_text_block("read_file", {"path": "x"})), False),
        ])
        _go(client, [{"role": "user", "content": "go"}], [READ], stream=True)
        correction = sdk.messages[1]["text"]
        assert "shell" in correction
        assert "tool-calling channel" in correction
        assert "ONLY by writing a <tool_call>" not in correction
    finally:
        client.close()


def test_09b_switch_off_miss_uses_event_name_and_upstream_wording(monkeypatch):
    _switch(monkeypatch, False)
    client = _client()
    try:
        sdk = client._sdk_client()
        repair = _text_block("read_file", {"path": "x"})
        _install(sdk, [
            ([_native("mcp", {"toolName": "shell", "args": {"cmd": "ls"}}, status="error", call_id="m1")], _result(), False),
            ([_assistant_event(repair)], _result(repair), False),
        ])
        _go(client, [{"role": "user", "content": "go"}], [READ], stream=True)
        correction = sdk.messages[1]["text"]
        assert "mcp" in correction
        assert "ONLY by writing a <tool_call>" in correction
    finally:
        client.close()


# --- 10 (C) ----------------------------------------------------------------


def test_10_text_tool_call_still_returned_when_switch_on(monkeypatch):
    _switch(monkeypatch, True)
    client = _client()
    try:
        sdk = client._sdk_client()
        block = _text_block("read_file", {"path": "x"}, "call_1")
        sdk.result.result = block
        sdk.result.stream_text = ""
        chunks = _go(client, [{"role": "user", "content": "go"}], [READ], stream=True)
        assert _names(chunks) == ["read_file"]
        assert json.loads(_stream_calls(chunks)[0].function.arguments) == {"path": "x"}
    finally:
        client.close()


# --- 11 (R) / 12 (R) -------------------------------------------------------


def _one_native(name, args, call_id="n1"):
    return ([_native(name, args, call_id=call_id, status="completed"), _status("FINISHED")], _result(), False)


def test_11_session_toolset_change_recreates_agent_and_resends(monkeypatch):
    _switch(monkeypatch, True)
    client = _client()
    try:
        sdk = client._sdk_client()
        _install(sdk, [
            _one_native("read_file", {"path": "a"}, "a1"),
            _one_native("read_file", {"path": "b"}, "a2"),
            _one_native("read_file", {"path": "c"}, "a3"),
            _one_native("read_file", {"path": "d"}, "a4"),
        ])
        first = _go(client, [{"role": "user", "content": "alpha-request-1"}], [READ], stream=True, scope="set-change")
        second_messages = _follow(
            "alpha-request-1", first,
            {"role": "tool", "tool_call_id": _stream_calls(first)[0].id, "content": "file-a"},
            {"role": "user", "content": "alpha-request-2"},
        )
        _go(client, second_messages, [READ, SEARCH], stream=True, scope="set-change")
        assert len(sdk.agents) == 2
        assert sdk.agents[0].closed
        assert set(sdk.options[1]["local"]["custom_tools"]) == {"read_file", "web_search"}
        assert "alpha-request-1" in sdk.messages[1]["text"]

        same = _go(client, [{"role": "user", "content": "beta-request-1"}], [READ, SEARCH], stream=True, scope="set-same")
        same_messages = _follow(
            "beta-request-1", same,
            {"role": "tool", "tool_call_id": _stream_calls(same)[0].id, "content": "file-b"},
            {"role": "user", "content": "beta-request-2"},
        )
        before = len(sdk.agents)
        _go(client, same_messages, [READ, SEARCH], stream=True, scope="set-same")
        assert len(sdk.agents) == before
        assert "beta-request-2" in sdk.messages[-1]["text"]
        assert "beta-request-1" not in sdk.messages[-1]["text"]
    finally:
        client.close()


def test_11b_description_only_change_recreates_with_toolset(monkeypatch, caplog):
    _switch(monkeypatch, True)
    caplog.set_level(logging.INFO, logger=LOGGER)
    client = _client()
    try:
        sdk = client._sdk_client()
        _install(sdk, [
            _one_native("read_file", {"path": "a"}, "d1"),
            _one_native("read_file", {"path": "b"}, "d2"),
        ])
        first = _go(
            client, [{"role": "user", "content": "desc-request-1"}], [READ],
            stream=True, scope="desc-only",
        )
        revised = _fn("read_file", "Read a different file", "path")
        follow = _follow(
            "desc-request-1", first,
            {"role": "tool", "tool_call_id": _stream_calls(first)[0].id, "content": "file-a"},
            {"role": "user", "content": "desc-request-2"},
        )
        _go(client, follow, [revised], stream=True, scope="desc-only")
        assert len(sdk.agents) == 2
        assert sdk.agents[0].closed
        assert sdk.options[1]["local"]["custom_tools"]["read_file"]["description"] == "Read a different file"
        assert "reason=toolset" in caplog.text
    finally:
        client.close()


def test_12_session_to_no_tools_retires_agent_even_if_switch_drops(monkeypatch):
    _switch(monkeypatch, True)
    client = _client()
    try:
        sdk = client._sdk_client()
        _install(sdk, [
            _one_native("read_file", {"path": "a"}, "r1"),
            ([_assistant_event("plain")], _result("plain"), False),
            _one_native("read_file", {"path": "b"}, "r2"),
            _one_native("read_file", {"path": "c"}, "r3"),
        ])
        first = _go(client, [{"role": "user", "content": "keep-user"}], [READ], stream=True, scope="drop-tools")
        follow = _follow(
            "keep-user", first,
            {"role": "tool", "tool_call_id": _stream_calls(first)[0].id, "content": "body"},
        )
        _go(client, follow, None, stream=True, scope="drop-tools")
        assert sdk.agents[0].closed
        assert "custom_tools" not in sdk.options[-1]["local"]

        other = _go(client, [{"role": "user", "content": "other-user"}], [READ], stream=True, scope="drop-switch")
        other_follow = _follow(
            "other-user", other,
            {"role": "tool", "tool_call_id": _stream_calls(other)[0].id, "content": "body"},
        )
        monkeypatch.delenv("HERMES_CURSOR_NATIVE_TOOLS", raising=False)
        _go(client, other_follow, [READ], stream=True, scope="drop-switch")
        assert sdk.agents[-2].closed
        assert "custom_tools" not in sdk.options[-1]["local"]
    finally:
        client.close()


# --- 13 --------------------------------------------------------------------


def _call_a():
    return _native("read_file", {"path": "x"}, call_id="A", status="completed")


def test_13a_status_error_after_calls_returns_them(monkeypatch, caplog):
    _switch(monkeypatch, True)
    caplog.set_level(logging.WARNING, logger=LOGGER)
    client = _client()
    try:
        sdk = client._sdk_client()
        script = [_call_a(), _thinking(), _native("web_search", {"q": "z"}, call_id="B"), _status("ERROR", "boom")]
        _install(sdk, [(script, _result(status="ERROR"), False)])
        chunks = _go(client, [{"role": "user", "content": "go"}], [READ, SEARCH], stream=True)
        assert _names(chunks) == ["read_file", "web_search"]
        assert "returning them" in caplog.text
    finally:
        client.close()


def test_13b_deadline_on_next_after_a_call_returns_it(monkeypatch, caplog):
    _switch(monkeypatch, True)
    caplog.set_level(logging.WARNING, logger=LOGGER)
    clock = Clock()
    monkeypatch.setattr(client_module.time, "monotonic", clock)
    client = _client()
    try:
        sdk = client._sdk_client()
        script = [_jump(clock, _call_a())]
        _install(sdk, [(script, _result(), False)])
        chunks = _go(client, [{"role": "user", "content": "go"}], [READ], stream=True)
        assert _names(chunks) == ["read_file"]
        assert "returning them" in caplog.text
    finally:
        client.close()


def test_13c_deadline_on_wait_after_a_call_returns_it(monkeypatch, caplog):
    _switch(monkeypatch, True)
    caplog.set_level(logging.WARNING, logger=LOGGER)
    clock = Clock()
    monkeypatch.setattr(client_module.time, "monotonic", clock)
    client = _client()
    try:
        sdk = client._sdk_client()
        def end_past_deadline(_run):
            clock.advance(10_000)
            return None

        script = [_call_a(), _status("FINISHED"), end_past_deadline]
        runs = _install(sdk, [(script, _result(), True)])
        chunks = _go(client, [{"role": "user", "content": "go"}], [READ], stream=True)
        assert _names(chunks) == ["read_file"]
        assert runs[0].wait_calls >= 1
        assert "returning them" in caplog.text
    finally:
        client.close()


def test_13d_cancelled_after_a_call_returns_it_with_warning(monkeypatch, caplog):
    _switch(monkeypatch, True)
    caplog.set_level(logging.INFO, logger=LOGGER)
    client = _client()
    try:
        sdk = client._sdk_client()
        script = [_call_a(), _thinking(), _status("CANCELLED", "remote")]
        _install(sdk, [(script, _result(), False)])
        chunks = _go(client, [{"role": "user", "content": "go"}], [READ], stream=True)
        assert _names(chunks) == ["read_file"]
        assert "returning them" in caplog.text
        assert "cancelled by" not in caplog.text
    finally:
        client.close()


def test_13e_zero_calls_status_error_raises(monkeypatch):
    _switch(monkeypatch, True)
    client = _client()
    try:
        sdk = client._sdk_client()
        _install(sdk, [([_status("ERROR", "nope")], _result(status="ERROR"), False)])
        with pytest.raises(CursorSDKRunError):
            _go(client, [{"role": "user", "content": "go"}], [READ], stream=True)
    finally:
        client.close()


def test_13f_wait_status_error_after_a_call_returns_it(monkeypatch, caplog):
    _switch(monkeypatch, True)
    caplog.set_level(logging.WARNING, logger=LOGGER)
    client = _client()
    try:
        sdk = client._sdk_client()
        runs = _install(sdk, [([_call_a(), _status("FINISHED")], _result(status="ERROR"), False)])
        chunks = _go(client, [{"role": "user", "content": "go"}], [READ], stream=True)
        assert _names(chunks) == ["read_file"]
        assert runs[0].wait_calls >= 1
        assert "returning them" in caplog.text
    finally:
        client.close()


def test_13g_zero_calls_deadline_raises_timeout(monkeypatch):
    _switch(monkeypatch, True)
    clock = Clock()
    monkeypatch.setattr(client_module.time, "monotonic", clock)
    client = _client()
    try:
        sdk = client._sdk_client()

        def create_agent(options):
            sdk.options.append(options)
            agent = FakeAgent(sdk)

            def send(message):
                clock.advance(10_000)
                agent.messages.append(message)
                sdk.messages.append(message)
                return OfferRun([_assistant_event("late")], _result("late"))

            agent.send = send
            sdk.agents.append(agent)
            return agent

        sdk.create_agent = create_agent
        with pytest.raises(CursorSDKRunError) as exc:
            _go(client, [{"role": "user", "content": "go"}], [READ], stream=True)
        assert exc.value.error_code == "cursor_run_timeout"
    finally:
        client.close()


def test_13h_sdk_error_after_a_call_returns_it_but_stopiteration_does_not(monkeypatch, caplog):
    _switch(monkeypatch, True)
    caplog.set_level(logging.WARNING, logger=LOGGER)
    client = _client()
    try:
        sdk = client._sdk_client()
        end = _install(sdk, [([_call_a()], _result("done"), False)])
        chunks = _go(client, [{"role": "user", "content": "end"}], [READ], stream=True, scope="end-ok")
        assert _names(chunks) == ["read_file"]
        assert end[0].wait_calls == 1
        assert "returning them" not in caplog.text

        boom = CursorSDKRunError("transport reset", error_code="cursor_sdk_run_failed")
        _install(sdk, [([_call_a(), boom], _result(), False)])
        chunks = _go(client, [{"role": "user", "content": "boom"}], [READ], stream=True, scope="boom")
        assert _names(chunks) == ["read_file"]
        assert "returning them" in caplog.text
    finally:
        client.close()


def test_13i_hermes_cancel_returns_calls_without_warning(monkeypatch, caplog):
    _switch(monkeypatch, True)
    caplog.set_level(logging.INFO, logger=LOGGER)
    client = _client()
    holder = {}
    try:
        sdk = client._sdk_client()

        def via_cancel(_run):
            client.cancel()
            return _status("CANCELLED", "hermes")

        def via_control(_run):
            holder["stream"]._control.closed = True
            return _status("CANCELLED", "closed")

        _install(sdk, [
            ([_call_a(), via_cancel], _result(), False),
            ([_call_a(), via_control], _result(), False),
        ])
        chunks = _go(client, [{"role": "user", "content": "c"}], [READ], stream=True, scope="cancel-hermes")
        assert _names(chunks) == ["read_file"]
        holder["stream"] = client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "d"}],
            tools=[READ],
            stream=True,
            _cursor_session_scope="cancel-control",
        )
        closed = list(holder["stream"])
        assert _names(closed) == ["read_file"]
        assert caplog.text.count("cancelled by hermes") >= 2
        assert "returning them" not in caplog.text
    finally:
        client.close()


def test_13j_wait_hermes_cancel_is_info(monkeypatch, caplog):
    _switch(monkeypatch, True)
    caplog.set_level(logging.INFO, logger=LOGGER)
    client = _client()
    try:
        sdk = client._sdk_client()

        def wait(self):
            self.wait_calls += 1
            client.cancel()
            return _result(status="CANCELLED")

        monkeypatch.setattr(OfferRun, "wait", wait)
        runs = _install(sdk, [([_call_a(), _status("FINISHED")], _result(), False)])
        chunks = _go(client, [{"role": "user", "content": "wait-cancel"}], [READ], stream=True)
        assert _names(chunks) == ["read_file"]
        assert runs[0].wait_calls >= 1
        assert "cancelled by hermes" in caplog.text
        assert "returning them" not in caplog.text
        assert not any(record.levelno >= logging.WARNING for record in caplog.records)
    finally:
        client.close()


def test_13k_wait_runtime_error_is_error_not_timeout(monkeypatch, caplog):
    _switch(monkeypatch, True)
    caplog.set_level(logging.WARNING, logger=LOGGER)
    client = _client()
    try:
        sdk = client._sdk_client()

        def wait(self):
            self.wait_calls += 1
            raise RuntimeError("sdk blew up")

        monkeypatch.setattr(OfferRun, "wait", wait)
        runs = _install(sdk, [([_call_a(), _status("FINISHED")], _result(), False)])
        chunks = _go(client, [{"role": "user", "content": "wait-error"}], [READ], stream=True)
        assert _names(chunks) == ["read_file"]
        assert runs[0].wait_calls >= 1
        assert "run ended error" in caplog.text
        assert "run ended timeout" not in caplog.text
    finally:
        client.close()


def test_13l_status_after_calls_cancels_run(monkeypatch):
    _switch(monkeypatch, True)
    client = _client()
    try:
        sdk = client._sdk_client()
        runs = _install(sdk, [
            ([_call_a(), _status("ERROR", "boom")], _result(status="ERROR"), False),
            ([_call_a(), _status("CANCELLED", "remote")], _result(status="CANCELLED"), False),
        ])
        error = _go(client, [{"role": "user", "content": "status-error"}], [READ], stream=True, scope="status-error")
        cancelled = _go(client, [{"role": "user", "content": "status-cancel"}], [READ], stream=True, scope="status-cancel")
        assert _names(error) == ["read_file"]
        assert _names(cancelled) == ["read_file"]
        assert runs[0].cancel_calls == 1
        assert runs[1].cancel_calls == 1
    finally:
        client.close()


# --- 14 (C) / 15 (R) / 16 (R) ----------------------------------------------


def test_14_required_choice_is_satisfied_by_native_call(monkeypatch):
    _switch(monkeypatch, True)
    client = _client()
    try:
        sdk = client._sdk_client()
        _install(sdk, [_one_native("read_file", {"path": "x"}, "req1")])
        chunks = _go(
            client, [{"role": "user", "content": "need"}], [READ],
            stream=True, tool_choice="required",
        )
        assert _names(chunks) == ["read_file"]
        assert len(sdk.messages) == 1
    finally:
        client.close()


def test_15_cap_returns_first_calls_and_cancels(monkeypatch):
    _switch(monkeypatch, True)
    monkeypatch.setenv("HERMES_CURSOR_NATIVE_MAX_CALLS", "2")
    client = _client()
    try:
        sdk = client._sdk_client()
        script = [
            _native("read_file", {"path": "a"}, call_id="1"),
            _native("read_file", {"path": "b"}, call_id="2"),
            _native("read_file", {"path": "c"}, call_id="3"),
        ]
        runs = _install(sdk, [(script, _result(), False)])
        chunks = _go(client, [{"role": "user", "content": "cap"}], [READ], stream=True)
        args = [json.loads(c.function.arguments)["path"] for c in _stream_calls(chunks)]
        assert args == ["a", "b"]
        assert runs[0].cancelled
    finally:
        client.close()


def test_16_tick_chunk_between_call_and_finish(monkeypatch):
    _switch(monkeypatch, True)
    clock = Clock()
    monkeypatch.setattr(client_module.time, "monotonic", clock)
    client = _client()
    try:
        sdk = client._sdk_client()
        script = [_call_a(), _jump(clock, _thinking()), _status("FINISHED")]
        # 31s, not a deadline jump: replace the 10_000 advance.
        def gap(_run):
            clock.advance(31)
            return _thinking()
        script[1] = gap
        _install(sdk, [(script, _result(), False)])
        chunks = _go(client, [{"role": "user", "content": "tick"}], [READ], stream=True)
        call_at = finish_at = None
        for index, chunk in enumerate(chunks):
            if not chunk.choices:
                continue
            if chunk.choices[0].delta.tool_calls:
                call_at = index
            if chunk.choices[0].finish_reason:
                finish_at = index
                break
        assert call_at is not None and finish_at is not None and finish_at > call_at + 1
        mid = chunks[call_at + 1:finish_at]
        assert any(
            item.choices
            and item.choices[0].delta.content is None
            and item.choices[0].delta.tool_calls is None
            and item.choices[0].finish_reason is None
            for item in mid
        )
    finally:
        client.close()


# --- 17 (R) ----------------------------------------------------------------


def test_17_logs_source_reason_and_suppresses_offered_warning(monkeypatch, caplog):
    _switch(monkeypatch, True)
    caplog.set_level(logging.INFO, logger=LOGGER)
    monkeypatch.setattr(client_module, "_NATIVE_INTERCEPT_FORCED_LOGGED", False, raising=False)
    client = _client()
    boom = {"armed": False}
    try:
        sdk = client._sdk_client()
        original = sdk.create_agent
        pending = []

        def create_agent(options):
            if boom["armed"]:
                boom["armed"] = False
                sdk.options.append(options)
                raise RuntimeError("boom")
            agent = original(options) if False else None
            sdk.options.append(options)
            agent = FakeAgent(sdk)

            def send(message):
                agent.messages.append(message)
                sdk.messages.append(message)
                if pending:
                    script, result, block = pending.pop(0)
                else:
                    script, result, block = ([_assistant_event("fallback")], _result("fallback"), False)
                return OfferRun(script, result, block_wait=block)

            agent.send = send
            sdk.agents.append(agent)
            return agent

        sdk.create_agent = create_agent
        secret = "SECRET_PATH_TOKEN"
        block = _text_block("web_search", {"q": "z"}, "text1")
        pending.append((
            [
                _native("read_file", {"path": secret}, call_id="log1"),
                _assistant_event("note " + block),
                _status("FINISHED"),
            ],
            _result(),
            False,
        ))
        _go(client, [{"role": "user", "content": "log-user"}], [READ, SEARCH], stream=True, scope="log-session")
        pending.append(([_assistant_event("ephemeral")], _result("ephemeral"), False))
        _go(client, [{"role": "user", "content": "no-tools"}], None, stream=True)

        pending.append(_one_native("read_file", {"path": "a"}, "t1"))
        first = _go(client, [{"role": "user", "content": "toolset-1"}], [READ], stream=True, scope="log-toolset")
        pending.append(_one_native("read_file", {"path": "b"}, "t2"))
        _go(client, _follow(
            "toolset-1", first,
            {"role": "tool", "tool_call_id": _stream_calls(first)[0].id, "content": "a"},
        ), [READ, SEARCH], stream=True, scope="log-toolset")

        pending.append(_one_native("read_file", {"path": "m"}, "m1"))
        mode = _go(client, [{"role": "user", "content": "mode-1"}], [READ], stream=True, scope="log-mode")
        pending.append(([_assistant_event("none")], _result("none"), False))
        _go(client, _follow(
            "mode-1", mode,
            {"role": "tool", "tool_call_id": _stream_calls(mode)[0].id, "content": "m"},
        ), None, stream=True, scope="log-mode")

        pending.append(([_assistant_event("hist")], _result("hist"), False))
        _go(client, [{"role": "user", "content": "hist-1"}], [READ], stream=True, scope="log-hist")
        pending.append(([_assistant_event("rewritten")], _result("rewritten"), False))
        _go(client, [{"role": "user", "content": "hist-rewritten"}], [READ], stream=True, scope="log-hist")

        pending.append((
            [_native("shell", {"cmd": "ls"}, status="error", call_id="bad"), _status("FINISHED")],
            _result(),
            False,
        ))
        pending.append(([_assistant_event("nope")], _result("nope"), False))
        _go(client, [{"role": "user", "content": "poison-1"}], [READ], stream=True, scope="log-poison")
        pending.append(([_assistant_event("after-poison")], _result("after-poison"), False))
        _go(client, [{"role": "user", "content": "poison-2"}], [READ], stream=True, scope="log-poison")

        pending.append(_one_native("read_file", {"path": "e"}, "e1"))
        exc_first = _go(client, [{"role": "user", "content": "exc-1"}], [READ], stream=True, scope="log-exc")
        boom["armed"] = True
        with pytest.raises(RuntimeError, match="boom"):
            _go(client, _follow(
                "exc-1", exc_first,
                {"role": "tool", "tool_call_id": _stream_calls(exc_first)[0].id, "content": "e"},
            ), [READ, SEARCH], stream=True, scope="log-exc")
        pending.append(([_assistant_event("after-exc")], _result("after-exc"), False))
        _go(client, [{"role": "user", "content": "exc-3"}], [READ], stream=True, scope="log-exc")

        assert secret not in caplog.text
        assert "source=native name=read_file" in caplog.text
        assert "source=text name=web_search" in caplog.text
        assert "native_tools=on tools=2 scope=session" in caplog.text
        assert "native_tools=off tools=0 scope=ephemeral" in caplog.text
        assert "reason=toolset" in caplog.text
        assert "reason=mode" in caplog.text
        assert "reason=history" in caplog.text
        assert "reason=poison" in caplog.text
        reasons = [line.rsplit("reason=", 1)[-1].split()[0] for line in caplog.text.splitlines() if "agent created" in line]
        assert reasons[-2:] == ["toolset", "new"]
        assert "intercepted native tool call" not in caplog.text

        monkeypatch.delenv("HERMES_CURSOR_NATIVE_TOOLS", raising=False)
        pending.append(([_native("read_file", {"path": "off"}, call_id="off1")], _result(), False))
        _go(client, [{"role": "user", "content": "off"}], [READ], stream=True)
        assert "intercepted native tool call" in caplog.text
    finally:
        client.close()


# --- 18 (C) ----------------------------------------------------------------


def test_18_image_is_forwarded_with_native_call(monkeypatch):
    _switch(monkeypatch, True)
    client = _client()
    try:
        sdk = client._sdk_client()
        _install(sdk, [_one_native("read_file", {"path": "x"}, "img1")])
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "see"},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
            ],
        }]
        chunks = _go(client, messages, [READ], stream=True)
        assert _names(chunks) == ["read_file"]
        assert sdk.messages[0]["images"] == [{"url": "https://example.com/a.png"}]
    finally:
        client.close()


# --- 19a (C) / 19b (R) -----------------------------------------------------


def test_19a_nonstream_stays_upstream(monkeypatch):
    _switch(monkeypatch, True)
    client = _client()
    try:
        sdk = client._sdk_client()
        block = _text_block("read_file", {"path": "x"}, "sync1")
        sdk.result.result = block
        sdk.result.stream_text = ""
        response = _go(
            client, FIXTURE_MESSAGES, FIXTURE_TOOLS, stream=False, scope="fresh-19a",
        )
        _assert_upstream_options(sdk.options[0])
        assert sdk.messages[0]["text"] == FIXTURE
        assert response.choices[0].message.tool_calls[0].function.name == "read_file"
        assert response.usage.prompt_tokens == sdk.result.usage.input_tokens
        assert response.usage.prompt_tokens > 0
    finally:
        client.close()


def test_19b_nonstream_in_native_session_retires_with_mode(monkeypatch, caplog):
    _switch(monkeypatch, True)
    caplog.set_level(logging.INFO, logger=LOGGER)
    client = _client()
    try:
        sdk = client._sdk_client()
        _install(sdk, [
            _one_native("read_file", {"path": "a"}, "s1"),
            ([_assistant_event("sync-answer")], _result("sync-answer"), False),
            _one_native("read_file", {"path": "c"}, "s3"),
        ])
        first = _go(client, [{"role": "user", "content": "scope-user-1"}], [READ], stream=True, scope="S")
        follow = _follow(
            "scope-user-1", first,
            {"role": "tool", "tool_call_id": _stream_calls(first)[0].id, "content": "file"},
            {"role": "user", "content": "scope-user-2"},
        )
        _go(client, follow, [READ], stream=False, scope="S")
        assert sdk.agents[0].closed
        assert "custom_tools" not in sdk.options[1]["local"]
        assert "reason=mode" in caplog.text
        assert "scope-user-1" in sdk.messages[1]["text"]
        third = _go(client, follow + [{"role": "user", "content": "scope-user-3"}], [READ], stream=True, scope="S")
        assert _names(third) == ["read_file"]
        assert "custom_tools" in sdk.options[-1]["local"]
        created = [line for line in caplog.text.splitlines() if "agent created" in line]
        assert created[-1].split("reason=", 1)[-1].startswith("mode")
        assert "reason=toolset" not in created[-1]
    finally:
        client.close()


# --- 20 --------------------------------------------------------------------


def test_20a_post_call_text_and_tail_are_calls(monkeypatch):
    _switch(monkeypatch, True)
    client = _client()
    try:
        sdk = client._sdk_client()
        event_block = _text_block("web_search", {"q": "event"}, "e1")
        tail_block = _text_block("web_search", {"q": "tail"}, "t1")
        _install(sdk, [
            ([
                _native("read_file", {"path": "x"}, call_id="n"),
                _assistant_event("note " + event_block + " done"),
                _status("FINISHED"),
            ], _result("note " + event_block + " done"), False),
            ([
                _native("read_file", {"path": "y"}, call_id="n2"),
                _status("FINISHED"),
            ], _result(tail_block), False),
        ])
        event_chunks = _go(client, [{"role": "user", "content": "event"}], [READ, SEARCH], stream=True, scope="post-event")
        assert _names(event_chunks) == ["read_file", "web_search"]
        assert _stream_text(event_chunks) == ""
        tail_chunks = _go(client, [{"role": "user", "content": "tail"}], [READ, SEARCH], stream=True, scope="post-tail")
        assert _names(tail_chunks) == ["read_file", "web_search"]
        assert json.loads(_stream_calls(tail_chunks)[1].function.arguments) == {"q": "tail"}
        assert _stream_text(tail_chunks) == ""
    finally:
        client.close()


def test_20b_running_call_at_normal_end_is_source_native(monkeypatch, caplog):
    _switch(monkeypatch, True)
    caplog.set_level(logging.INFO, logger=LOGGER)
    client = _client()
    try:
        sdk = client._sdk_client()
        script = [_native("read_file", {"path": "x"}, status="running", call_id="run1")]
        runs = _install(sdk, [(script, _result("done"), False)])
        chunks = _go(client, [{"role": "user", "content": "end"}], [READ], stream=True)
        assert _names(chunks) == ["read_file"]
        assert "source=native name=read_file" in caplog.text
        assert "unfinished native tool call" not in caplog.text
        assert runs[0].wait_calls == 1
    finally:
        client.close()


def test_20c_flush_does_not_leak_partial_opener(monkeypatch):
    _switch(monkeypatch, True)
    client = _client()
    try:
        sdk = client._sdk_client()
        script = [
            _native("read_file", {"path": "x"}, call_id="p1"),
            _assistant_event("looking <tool"),
            _status("FINISHED"),
        ]
        runs = _install(sdk, [(script, _result("looking <tool"), False)])
        chunks = _go(client, [{"role": "user", "content": "partial"}], [READ], stream=True)
        assert _names(chunks) == ["read_file"]
        assert not runs[0].cancelled
        assert not any(
            chunk.choices and (
                (chunk.choices[0].delta.content and "<tool" in chunk.choices[0].delta.content)
                or chunk.choices[0].delta.tool_calls and False
            )
            for chunk in chunks
        )
        assert not any(
            chunk.choices and getattr(chunk.choices[0].delta, "content", None) and "rejected" in str(chunk.choices[0].delta.content)
            for chunk in chunks
        )
        textish = [
            chunk for chunk in chunks
            if chunk.choices and chunk.choices[0].delta.content
        ]
        assert textish == []
    finally:
        client.close()


def test_20c_prime_quiet_unusable_block(monkeypatch, caplog):
    _switch(monkeypatch, True)
    caplog.set_level(logging.WARNING, logger=LOGGER)
    client = _client()
    try:
        sdk = client._sdk_client()
        junk = "<tool_call>{not json}</tool_call>"
        script = [
            _native("read_file", {"path": "x"}, call_id="q1"),
            _assistant_event("before " + junk + " after"),
            _status("FINISHED"),
        ]
        runs = _install(sdk, [(script, _result("before " + junk + " after"), False)])
        chunks = _go(client, [{"role": "user", "content": "quiet"}], [READ], stream=True)
        assert _names(chunks) == ["read_file"]
        assert not runs[0].cancelled
        assert _stream_text(chunks) == ""
        assert "rejected an unusable" not in caplog.text
    finally:
        client.close()


def test_20d_running_call_not_accepted_after_cap(monkeypatch):
    _switch(monkeypatch, True)
    monkeypatch.setenv("HERMES_CURSOR_NATIVE_MAX_CALLS", "2")
    client = _client()
    try:
        sdk = client._sdk_client()
        script = [
            _native("read_file", {"path": "a"}, call_id="1"),
            _native("web_search", {"q": "pending"}, call_id="running", status="running"),
            _native("read_file", {"path": "b"}, call_id="2"),
            _native("read_file", {"path": "c"}, call_id="3"),
        ]
        runs = _install(sdk, [(script, _result(), False)])
        chunks = _go(client, [{"role": "user", "content": "cut"}], [READ, SEARCH], stream=True)
        args = [json.loads(c.function.arguments) for c in _stream_calls(chunks)]
        assert args == [{"path": "a"}, {"path": "b"}]
        assert runs[0].cancelled
        assert not any(item.get("q") == "pending" for item in args)
    finally:
        client.close()


def test_20e_text_call_counts_toward_cap(monkeypatch, caplog):
    _switch(monkeypatch, True)
    monkeypatch.setenv("HERMES_CURSOR_NATIVE_MAX_CALLS", "2")
    caplog.set_level(logging.INFO, logger=LOGGER)
    client = _client()
    try:
        sdk = client._sdk_client()
        block = _text_block("web_search", {"q": "b"}, "tb")
        script = [
            _native("read_file", {"path": "a"}, call_id="a"),
            _assistant_event(block),
            _native("read_file", {"path": "c"}, call_id="c"),
        ]
        runs = _install(sdk, [(script, _result(), False)])
        chunks = _go(client, [{"role": "user", "content": "mid"}], [READ, SEARCH], stream=True)
        assert _names(chunks) == ["read_file", "web_search"]
        assert runs[0].cancelled
        assert caplog.text.count("cap reached") == 1
    finally:
        client.close()


def test_20f_tail_calls_past_cap_are_dropped(monkeypatch, caplog):
    _switch(monkeypatch, True)
    monkeypatch.setenv("HERMES_CURSOR_NATIVE_MAX_CALLS", "2")
    caplog.set_level(logging.INFO, logger=LOGGER)
    client = _client()
    try:
        sdk = client._sdk_client()
        tail = _text_block("web_search", {"q": "b"}, "tb") + _text_block("web_search", {"q": "c"}, "tc")
        script = [_native("read_file", {"path": "a"}, call_id="a"), _status("FINISHED")]
        _install(sdk, [(script, _result(tail), False)])
        chunks = _go(client, [{"role": "user", "content": "tailcap"}], [READ, SEARCH], stream=True)
        assert [json.loads(c.function.arguments) for c in _stream_calls(chunks)] == [
            {"path": "a"},
            {"q": "b"},
        ]
        assert caplog.text.count("cap reached") == 1
    finally:
        client.close()


# --- 21 (R) / 22 (R) -------------------------------------------------------


def test_21a_forced_intercept_logged_once(monkeypatch, caplog):
    _switch(monkeypatch, True)
    monkeypatch.setenv("HERMES_CURSOR_INTERCEPT_NATIVE", "0")
    monkeypatch.setattr(client_module, "_NATIVE_INTERCEPT_FORCED_LOGGED", False, raising=False)
    caplog.set_level(logging.INFO, logger=LOGGER)
    client = _client()
    try:
        sdk = client._sdk_client()
        _install(sdk, [
            _one_native("read_file", {"path": "a"}, "f1"),
            _one_native("read_file", {"path": "b"}, "f2"),
        ])
        _go(client, [{"role": "user", "content": "one"}], [READ], stream=True)
        _go(client, [{"role": "user", "content": "two"}], [READ], stream=True, scope="forced-2")
        assert caplog.text.count("interception forced on") == 1
    finally:
        client.close()


def test_21b_newline_only_id_falls_back(monkeypatch):
    _switch(monkeypatch, True)
    client = _client()
    try:
        sdk = client._sdk_client()
        _install(sdk, [([_native("read_file", {"path": "x"}, call_id="\n")], _result(), False)])
        chunks = _go(client, [{"role": "user", "content": "id"}], [READ], stream=True)
        assert _stream_calls(chunks)[0].id == "cursor_native_1"
    finally:
        client.close()


def test_21c_cap_clamps_zero_to_one_and_large_to_1024(monkeypatch):
    _switch(monkeypatch, True)
    client = _client()
    try:
        sdk = client._sdk_client()
        monkeypatch.setenv("HERMES_CURSOR_NATIVE_MAX_CALLS", "0")
        two = [
            _native("read_file", {"path": "a"}, call_id="1"),
            _native("read_file", {"path": "b"}, call_id="2"),
        ]
        runs = _install(sdk, [(two, _result(), False)])
        chunks = _go(client, [{"role": "user", "content": "zero"}], [READ], stream=True)
        assert [json.loads(c.function.arguments)["path"] for c in _stream_calls(chunks)] == ["a"]
        assert runs[0].cancelled

        monkeypatch.setenv("HERMES_CURSOR_NATIVE_MAX_CALLS", "5000")
        many = [_native("read_file", {"path": str(i)}, call_id=f"id{i}") for i in range(1025)]
        runs = _install(sdk, [(many, _result(), False)])
        chunks = _go(client, [{"role": "user", "content": "big"}], [READ], stream=True, scope="clamp")
        got = [json.loads(c.function.arguments)["path"] for c in _stream_calls(chunks)]
        assert got == [str(i) for i in range(1024)]
        assert runs[-1].cancelled
    finally:
        client.close()


def test_22_named_tool_choice_recreates_with_that_tool_only(monkeypatch):
    _switch(monkeypatch, True)
    client = _client()
    try:
        sdk = client._sdk_client()
        _install(sdk, [
            _one_native("read_file", {"path": "a"}, "c1"),
            _one_native("read_file", {"path": "b"}, "c2"),
        ])
        first = _go(client, [{"role": "user", "content": "both"}], [READ, SEARCH], stream=True, scope="named")
        follow = _follow(
            "both", first,
            {"role": "tool", "tool_call_id": _stream_calls(first)[0].id, "content": "a"},
        )
        _go(
            client, follow, [READ, SEARCH], stream=True, scope="named",
            tool_choice={"type": "function", "function": {"name": "read_file"}},
        )
        assert sdk.agents[0].closed
        assert set(sdk.options[-1]["local"]["custom_tools"]) == {"read_file"}
    finally:
        client.close()


# --- 23 (R) N7 per-request native_tools ------------------------------------


_FAST = {"id": "grok-4.7", "params": [{"id": "fast", "value": "true"}]}


def _assert_name_on(options, prompt):
    tools = options["local"]["custom_tools"]
    assert set(tools) == {"read_file", "web_search"}
    assert tools["read_file"]["input_schema"] == READ["function"]["parameters"]
    assert options["tools"] == ["mcp"]
    assert options["mcp_servers"] == {}
    assert options["model"] == _FAST
    assert "native_tools" not in json.dumps(options["model"])
    assert "The only tools you have are the Hermes tools listed below." in prompt
    assert "Call Hermes tools through your tool-calling channel" in prompt
    assert "MCP servers are disabled" not in prompt
    assert prompt != FIXTURE


def test_23a_model_on_offers_tools_and_beats_env_off(monkeypatch):
    monkeypatch.delenv("HERMES_CURSOR_NATIVE_TOOLS", raising=False)
    client = _client()
    try:
        sdk = client._sdk_client()
        _go(
            client, FIXTURE_MESSAGES, FIXTURE_TOOLS, stream=True,
            model="grok-4.7@fast=true,native_tools=on",
        )
        _assert_name_on(sdk.options[0], sdk.messages[0]["text"])
    finally:
        client.close()

    monkeypatch.setenv("HERMES_CURSOR_NATIVE_TOOLS", "0")
    client = _client()
    try:
        sdk = client._sdk_client()
        _go(
            client, FIXTURE_MESSAGES, FIXTURE_TOOLS, stream=True,
            model="grok-4.7@fast=true,native_tools=on", scope="23a-env-off",
        )
        _assert_name_on(sdk.options[0], sdk.messages[0]["text"])
    finally:
        client.close()


def test_23b_model_off_beats_env_on(monkeypatch):
    monkeypatch.setenv("HERMES_CURSOR_NATIVE_TOOLS", "1")
    client = _client()
    try:
        sdk = client._sdk_client()
        _go(
            client, FIXTURE_MESSAGES, FIXTURE_TOOLS, stream=True,
            model="grok-4.7@native_tools=off",
        )
        _assert_upstream_options(sdk.options[0])
        assert sdk.options[0].get("model") == "grok-4.7"
        assert sdk.messages[0]["text"] == FIXTURE
    finally:
        client.close()


def test_23c_bad_native_tools_fails_before_bridge_or_agent(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("must not be called")

    monkeypatch.setattr(client_module, "_bridge_for", boom)
    monkeypatch.setattr(CursorSDKClient, "_create_agent", boom)
    client = _client()
    try:
        with pytest.raises(ValueError, match=r"native_tools.*must be on or off"):
            list(client.chat.completions.create(
                model="grok-4.7@native_tools=maybe",
                messages=FIXTURE_MESSAGES,
                tools=FIXTURE_TOOLS,
                stream=True,
            ))
    finally:
        client.close()


def test_23d_only_native_tools_is_bare_id_without_catalog(monkeypatch):
    calls = []
    original = FakeCursorClient.list_models

    def wrapped(self, *, api_key):
        calls.append(api_key)
        return original(self, api_key=api_key)

    monkeypatch.setattr(FakeCursorClient, "list_models", wrapped)
    client = _client()
    try:
        sdk = client._sdk_client()
        _go(
            client, FIXTURE_MESSAGES, FIXTURE_TOOLS, stream=True,
            model="grok-4.7@native_tools=on",
        )
        assert sdk.options[0]["model"] == "grok-4.7"
        assert calls == []
    finally:
        client.close()


def test_23e_listing_keeps_native_tools_when_catalog_has_only_fast(monkeypatch):
    monkeypatch.setenv(
        "HERMES_CURSOR_MODEL_VARIANTS",
        "grok-4.7@fast=true,native_tools=on;grok-4.7@native_tools=on",
    )
    monkeypatch.setattr(
        FakeCursorClient,
        "models",
        [SimpleNamespace(
            id="grok-4.7",
            parameters=[_parameter("fast", "true", "false")],
            variants=[],
        )],
    )
    listed = client_module.list_cursor_models(api_key="crsr_test")
    assert "grok-4.7@fast=true,native_tools=on" in listed
    assert "grok-4.7@native_tools=on" in listed


def test_23f_native_tools_stays_in_session_key_and_splits_agents():
    assert client_module._model_id("grok-4.7@native_tools=on") != client_module._model_id(
        "grok-4.7@native_tools=off"
    )
    client = _client()
    try:
        sdk = client._sdk_client()
        sdk.result.stream_text = "hello"
        sdk.result.result = "hello"
        first = _go(
            client, [{"role": "user", "content": "ping"}], None, stream=True,
            scope="scope-23f", model="grok-4.7@native_tools=on",
        )
        assert _stream_text(first) == "hello"
        _go(
            client,
            [
                {"role": "user", "content": "ping"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "again"},
            ],
            None,
            stream=True,
            scope="scope-23f",
            model="grok-4.7@native_tools=off",
        )
        assert len(sdk.agents) == 2
    finally:
        client.close()


def test_23g_model_on_nonstream_stays_upstream(monkeypatch):
    monkeypatch.delenv("HERMES_CURSOR_NATIVE_TOOLS", raising=False)
    client = _client()
    try:
        sdk = client._sdk_client()
        _go(
            client, FIXTURE_MESSAGES, FIXTURE_TOOLS, stream=False, scope="fresh-23g",
            model="grok-4.7@native_tools=on",
        )
        _assert_upstream_options(sdk.options[0])
        assert sdk.options[0].get("model") == "grok-4.7"
        assert sdk.messages[0]["text"] == FIXTURE
    finally:
        client.close()


# --- 24a–24k (N8 second-answer cut) ----------------------------------------

NATIVE_MODEL = "grok-4.7@native_tools=on"
# da45a58, second 24c case (title, usage, assistant "\\n", then the call): the
# stream layer holds the whitespace chunk and never emits it. Pinned literal.
_24C_WHITESPACE_CONTENT = '{"title": "x"}'


def _n8_lines(caplog):
    return [line for line in caplog.text.splitlines() if "run ended on second answer" in line]


def _created_reason(caplog, reason):
    return [
        line for line in caplog.text.splitlines()
        if "agent created" in line and f"reason={reason}" in line
    ]


def test_24a_second_answer_is_cut(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    client = _client()
    try:
        sdk = client._sdk_client()
        script = [
            _assistant_event("Answer A"),
            _usage_event(10),
            _assistant_event("Answer B"),
            _usage_event(12),
            _status("FINISHED"),
        ]
        runs = _install(sdk, [(script, _result("Answer AAnswer B", input_tokens=99), False)])
        chunks = _go(
            client, [{"role": "user", "content": "n8-24a-user"}], [READ],
            stream=True, model=NATIVE_MODEL,
        )
        assert _stream_text(chunks) == "Answer A"
        assert runs[0].cancel_calls == 1
        assert runs[0].wait_calls == 0
        assert chunks[-1].usage.prompt_tokens == 10
        assert caplog.text.count("run ended on second answer after 0 native calls") == 1
    finally:
        client.close()


def test_24b_call_turn_is_not_a_second_answer(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    client = _client()
    try:
        sdk = client._sdk_client()
        script = [
            _assistant_event("Checking."),
            _native("read_file", {"path": "x"}, status="running", call_id="b1"),
            _native("read_file", {"path": "x"}, status="completed", call_id="b1"),
            _assistant_event("Done."),
            _usage_event(10),
            _status("FINISHED"),
        ]
        runs = _install(sdk, [(script, _result(input_tokens=99), False)])
        chunks = _go(
            client, [{"role": "user", "content": "n8-24b-user"}], [READ],
            stream=True, model=NATIVE_MODEL,
        )
        assert _names(chunks) == ["read_file"]
        assert runs[0].wait_calls == 1
        assert runs[0].cancel_calls == 0
        assert _n8_lines(caplog) == []
    finally:
        client.close()


def _24c_script(whitespace):
    script = [
        _assistant_event('{"title": "x"}'),
        _usage_event(5),
    ]
    if whitespace:
        script.append(_assistant_event("\n"))
    script.extend([
        _native("read_file", {"path": "x"}, status="running", call_id="c24"),
        _native("read_file", {"path": "x"}, status="completed", call_id="c24"),
        _assistant_event("ok"),
        _usage_event(7),
        _status("FINISHED"),
    ])
    return script


def _assert_24c_survives(chunks, run, caplog):
    assert _names(chunks) == ["read_file"]
    assert run.wait_calls == 1
    assert run.cancel_calls == 0
    assert _n8_lines(caplog) == []


def test_24c_title_preamble_then_call_is_not_cut(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    client = _client()
    try:
        sdk = client._sdk_client()
        runs = _install(sdk, [(_24c_script(False), _result(input_tokens=99), False)])
        chunks = _go(
            client, [{"role": "user", "content": "n8-24c-user"}], [READ],
            stream=True, model=NATIVE_MODEL,
        )
        _assert_24c_survives(chunks, runs[0], caplog)
    finally:
        client.close()


def test_24c_whitespace_chunk_does_not_cut(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    client = _client()
    try:
        sdk = client._sdk_client()
        runs = _install(sdk, [(_24c_script(True), _result(input_tokens=99), False)])
        chunks = _go(
            client, [{"role": "user", "content": "n8-24c-user"}], [READ],
            stream=True, model=NATIVE_MODEL,
        )
        _assert_24c_survives(chunks, runs[0], caplog)
        assert _stream_text(chunks) == _24C_WHITESPACE_CONTENT
    finally:
        client.close()


def test_24d_native_off_keeps_both_answers():
    client = _client()
    try:
        sdk = client._sdk_client()
        script = [
            _assistant_event("Answer A"),
            _usage_event(10),
            _assistant_event("Answer B"),
            _usage_event(12),
            _status("FINISHED"),
        ]
        runs = _install(sdk, [(script, _result("Answer AAnswer B", input_tokens=99), False)])
        chunks = _go(
            client, [{"role": "user", "content": "n8-24d-user"}], [READ],
            stream=True, model="grok-4.7@native_tools=off",
        )
        assert _stream_text(chunks) == "Answer AAnswer B"
        assert runs[0].wait_calls == 1
        assert runs[0].cancel_calls == 0
    finally:
        client.close()


def test_24e_pending_call_blocks_the_cut():
    client = _client()
    try:
        sdk = client._sdk_client()
        script = [
            _assistant_event("Reading."),
            _native("read_file", {"path": "x"}, status="running", call_id="pend"),
            _usage_event(10),
            _assistant_event("Aside."),
            _usage_event(11),
            _assistant_event("More."),
            _native("read_file", {"path": "x"}, status="completed", call_id="pend"),
            _status("FINISHED"),
        ]
        runs = _install(sdk, [(script, _result(input_tokens=99), False)])
        chunks = _go(
            client, [{"role": "user", "content": "n8-24e-user"}], [READ],
            stream=True, model=NATIVE_MODEL,
        )
        assert runs[0].cancel_calls == 0
        assert _names(chunks) == ["read_file"]
    finally:
        client.close()


def test_24f_nonstream_stays_upstream(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    client = _client()
    try:
        sdk = client._sdk_client()
        _go(
            client, FIXTURE_MESSAGES, FIXTURE_TOOLS, stream=False, scope="n8-24f",
            model=NATIVE_MODEL,
        )
        _assert_upstream_options(sdk.options[0])
        assert sdk.messages[0]["text"] == FIXTURE
        assert _n8_lines(caplog) == []
    finally:
        client.close()


def test_24g_thinking_between_usages_does_not_cut(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    client = _client()
    try:
        sdk = client._sdk_client()
        script = [
            _assistant_event("Look."),
            _native("read_file", {"path": "x"}, status="running", call_id="g1"),
            _native("read_file", {"path": "x"}, status="completed", call_id="g1"),
            _usage_event(10),
            _thinking(),
            _usage_event(5),
            _assistant_event("Final."),
            _status("FINISHED"),
        ]
        runs = _install(sdk, [(script, _result(input_tokens=99), False)])
        chunks = _go(
            client, [{"role": "user", "content": "n8-24g-user"}], [READ],
            stream=True, model=NATIVE_MODEL,
        )
        assert runs[0].cancel_calls == 0
        assert runs[0].wait_calls == 1
        assert _n8_lines(caplog) == []
        assert _names(chunks) == ["read_file"]
    finally:
        client.close()


def test_24h_billed_is_the_sum_to_the_cut():
    client = _client()
    try:
        sdk = client._sdk_client()
        script = [
            _thinking(),
            _usage_event(5),
            _assistant_event("Answer"),
            _usage_event(7),
            _assistant_event("Again"),
        ]
        _install(sdk, [(script, _result(input_tokens=99), False)])
        chunks = _go(
            client, [{"role": "user", "content": "n8-24h-user"}], [READ],
            stream=True, model=NATIVE_MODEL,
        )
        assert _stream_text(chunks) == "Answer"
        assert chunks[-1].usage.prompt_tokens == 12
    finally:
        client.close()


def test_24i_calls_path_cut_bills_the_sum_and_anchors_the_peak(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    client = _client()
    try:
        sdk = client._sdk_client()
        script = [
            _assistant_event("Look."),
            _native("read_file", {"path": "x"}, status="running", call_id="i1"),
            _native("read_file", {"path": "x"}, status="completed", call_id="i1"),
            _usage_event(10),
            _assistant_event("Text one."),
            _usage_event(6),
            _assistant_event("Text two."),
        ]
        runs = _install(sdk, [(script, _result(input_tokens=99), False)])
        chunks = _go(
            client, [{"role": "user", "content": "n8-24i-user"}], [READ],
            stream=True, model=NATIVE_MODEL,
        )
        assert runs[0].cancel_calls == 1
        assert _names(chunks) == ["read_file"]
        assert chunks[-1].usage.prompt_tokens == 16
        assert chunks[-1].usage.initial_prompt_tokens == 10
        assert caplog.text.count("run ended on second answer after 1 native calls") == 1
    finally:
        client.close()


def test_24j_no_call_cut_retires_agent_and_resends(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    client = _client()
    try:
        sdk = client._sdk_client()
        user = "n8-24a-user-text"
        follow = "n8-24j-new-user"
        first_script = [
            _assistant_event("Answer A"),
            _usage_event(10),
            _assistant_event("Answer B"),
            _usage_event(12),
            _status("FINISHED"),
        ]
        _install(sdk, [
            (first_script, _result("Answer AAnswer B", input_tokens=99), False),
            ([_assistant_event("Follow ok")], _result("Follow ok"), False),
        ])
        first = _go(
            client, [{"role": "user", "content": user}], [READ],
            stream=True, scope="n8-24j", model=NATIVE_MODEL,
        )
        second_messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": _stream_text(first)},
            {"role": "user", "content": follow},
        ]
        second = _go(
            client, second_messages, [READ],
            stream=True, scope="n8-24j", model=NATIVE_MODEL,
        )
        assert len(_created_reason(caplog, "cut")) == 1
        sent = sdk.messages[-1]["text"]
        assert user in sent
        assert "Answer A" in sent
        assert follow in sent
        assert _stream_text(second) == "Follow ok"
    finally:
        client.close()


def test_24k_calls_cut_retires_agent_and_resends(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    client = _client()
    try:
        sdk = client._sdk_client()
        user = "n8-24i-user-text"
        script = [
            _assistant_event("Look."),
            _native("read_file", {"path": "x"}, status="running", call_id="k1"),
            _native("read_file", {"path": "x"}, status="completed", call_id="k1"),
            _usage_event(10),
            _assistant_event("Text one."),
            _usage_event(6),
            _assistant_event("Text two."),
        ]
        _install(sdk, [
            (script, _result(input_tokens=99), False),
            ([_assistant_event("k-follow")], _result("k-follow"), False),
        ])
        first = _go(
            client, [{"role": "user", "content": user}], [READ],
            stream=True, scope="n8-24k", model=NATIVE_MODEL,
        )
        second_messages = _follow(
            user, first,
            {"role": "tool", "tool_call_id": _stream_calls(first)[0].id, "content": "file-k"},
        )
        second = _go(
            client, second_messages, [READ],
            stream=True, scope="n8-24k", model=NATIVE_MODEL,
        )
        assert len(_created_reason(caplog, "cut")) == 1
        assert user in sdk.messages[-1]["text"]
        assert _stream_text(second) == "k-follow"
    finally:
        client.close()
