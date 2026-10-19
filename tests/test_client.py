from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
import logging
from pathlib import Path
import re
import threading
import time
from types import SimpleNamespace

import pytest

from hermes_cursor_provider import client as client_module
from hermes_cursor_provider.client import (
    CURSOR_API_URL,
    CursorSDKClient,
    extract_images,
    extract_tool_calls,
    list_cursor_models,
    render_prompt,
)


@dataclass
class FakeResult:
    result: str
    model: object
    usage: object
    stream_text: str = ""
    status: str = "FINISHED"
    status_message: str = ""


def _assistant_event(text):
    return SimpleNamespace(
        type="assistant",
        message=SimpleNamespace(content=[SimpleNamespace(text=text)]),
    )


def _status_event(status, message=""):
    return SimpleNamespace(type="status", status=status, message=message)


class FakeRun:
    def __init__(self, result):
        self._result = result
        self.cancelled = False

    def wait(self):
        return self._result

    def stream(self):
        if self._result.stream_text:
            yield _assistant_event(self._result.stream_text)
        if self._result.status != "FINISHED":
            yield _status_event(self._result.status, self._result.status_message)

    def cancel(self):
        self.cancelled = True


class ScriptedRun(FakeRun):
    """FakeRun with a scripted stream; callables run during iteration and may block."""

    def __init__(self, result, script):
        super().__init__(result)
        self.script = list(script)
        self.cancel_event = threading.Event()

    def stream(self):
        for item in self.script:
            if callable(item):
                item = item(self)
            if item is not None:
                yield item

    def cancel(self):
        super().cancel()
        self.cancel_event.set()


class FailingRun(FakeRun):
    def wait(self):
        raise TimeoutError("run timed out")


class ToolCallbackRun(FakeRun):
    """Fires the registered custom tool mid-stream, as the SDK's loopback callback does.

    The real SDK calls ``local.custom_tools[name].execute`` while the run is streaming
    and emits a ``tool_call`` event carrying the same id; both halves are faked here so
    the sink entry and the run's event ids can be reconciled.
    """

    def __init__(self, owner, name):
        super().__init__(owner.result)
        self.owner = owner
        self.name = name
        self.callback_result = None

    def stream(self):
        options = self.owner.options[-1] if self.owner.options else {}
        tools = (options.get("local") or {}).get("custom_tools") or {}
        entry = tools.get(self.name)
        if entry is not None:
            self.callback_result = entry["execute"](
                {"token": "probe-1"}, SimpleNamespace(tool_call_id="call_probe")
            )
            self.owner.tool_callbacks.append(self.callback_result)
            yield SimpleNamespace(type="tool_call", call_id="call_probe\nfc_0")
        yield from super().stream()


class FakeAgent:
    def __init__(self, owner):
        self.owner = owner
        self.closed = False
        self.messages = []

    def send(self, message):
        self.owner.messages.append(message)
        self.messages.append(message)
        # Only the newest transcript entry may request a call: the rendered prompt
        # replays earlier user turns, and those must not re-trigger the callback.
        tail = str((message or {}).get("text", "")).rsplit('{"role":"user"', 1)[-1]
        match = re.search(r"CALL\s+(\w+)", tail)
        if match:
            return ToolCallbackRun(self.owner, match.group(1))
        return FakeRun(self.owner.result)

    def close(self):
        self.closed = True


class FakeCursorClient:
    launches = []
    instances = []
    models = [SimpleNamespace(id="composer-2.5"), SimpleNamespace(id="gpt-5.4")]

    def __init__(self):
        self.options = []
        self.messages = []
        self.agents = []
        self.tool_callbacks = []
        self.closed = False
        self.result = FakeResult(
            result="hello",
            model=SimpleNamespace(id="composer-2.5"),
            usage=SimpleNamespace(
                input_tokens=4,
                output_tokens=2,
                cache_read_tokens=1,
                cache_write_tokens=2,
                total_tokens=9,
            ),
        )
        self.__class__.instances.append(self)

    @classmethod
    def launch_bridge(cls, **kwargs):
        cls.launches.append(kwargs)
        return cls()

    def with_options(self, **kwargs):
        self.timeout_options = kwargs
        return self

    def create_agent(self, options):
        self.options.append(options)
        agent = FakeAgent(self)
        self.agents.append(agent)
        return agent

    def list_models(self, *, api_key):
        self.list_api_key = api_key
        return self.models

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def fake_sdk(monkeypatch, tmp_path):
    client_module._reset_session_registry()
    FakeCursorClient.launches.clear()
    FakeCursorClient.instances.clear()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    module = SimpleNamespace(CursorClient=FakeCursorClient)
    monkeypatch.setattr(client_module, "importlib", SimpleNamespace(import_module=lambda name: module))
    yield
    client_module._reset_session_registry()


def _tool(name="read_file"):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Read one file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }


def _stream_text(chunks):
    return "".join(
        chunk.choices[0].delta.content or ""
        for chunk in chunks
        if chunk.choices and chunk.choices[0].delta.content
    )


def _stream_calls(chunks):
    return [
        call
        for chunk in chunks
        if chunk.choices and chunk.choices[0].delta.tool_calls
        for call in chunk.choices[0].delta.tool_calls
    ]


def _first_chunk_index(chunks, predicate):
    return next(
        index
        for index, chunk in enumerate(chunks)
        if chunk.choices and predicate(chunk.choices[0])
    )


def _assert_held_text_precedes_call_deltas(chunks):
    """Order at run end: held text -> tool-call deltas -> finish chunk."""
    text = _first_chunk_index(chunks, lambda choice: bool(choice.delta.content))
    call = _first_chunk_index(chunks, lambda choice: bool(choice.delta.tool_calls))
    finish = _first_chunk_index(chunks, lambda choice: bool(choice.finish_reason))
    assert text < call < finish


def _scripted_run(sdk, script):
    run = ScriptedRun(sdk.result, script)
    agent = FakeAgent(sdk)
    agent.send = lambda message: run
    sdk.create_agent = lambda options: agent
    return run


def _collect_scripted(client, sdk, text, split, tools_list=None):
    sdk.result.result = text
    sdk.result.stream_text = ""
    run = _scripted_run(sdk, [_assistant_event(text[:split]), _assistant_event(text[split:])])
    chunks = list(
        client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "hi"}],
            tools=tools_list if tools_list is not None else [_tool()],
            stream=True,
        )
    )
    return run, _stream_calls(chunks), _stream_text(chunks)


def test_prompt_keeps_hermes_authoritative_and_filters_tool_choice():
    prompt, allowed = render_prompt(
        [{"role": "user", "content": "inspect it"}],
        tools=[_tool(), _tool("write_file")],
        tool_choice={"type": "function", "function": {"name": "read_file"}},
    )
    assert allowed == {"read_file"}
    assert "Hermes is the only tool control plane" in prompt
    assert '"name":"read_file"' in prompt
    assert '"name":"write_file"' not in prompt
    assert render_prompt([], tools=[_tool()], tool_choice="none")[1] == set()


def test_native_prompt_drops_the_text_protocol():
    prompt, allowed = render_prompt(
        [{"role": "user", "content": "inspect it"}],
        tools=[_tool()],
        native_tools=True,
    )
    assert allowed == {"read_file"}
    assert "offered as native tool calls" in prompt
    assert "call it directly with the native tool-call mechanism" in prompt
    assert "<tool_call>" not in prompt
    assert '"name":"read_file"' in prompt

    # The default (no native tools) still instructs the <tool_call> block protocol.
    text_prompt, _ = render_prompt([{"role": "user", "content": "inspect it"}], tools=[_tool()])
    assert "<tool_call>" in text_prompt
    assert "offered as native tool calls" not in text_prompt

    # tool_choice="none" offers no tools at all, so neither protocol is advertised.
    none_prompt, none_allowed = render_prompt(
        [{"role": "user", "content": "inspect it"}],
        tools=[_tool()],
        tool_choice="none",
        native_tools=True,
    )
    assert none_allowed == set()
    assert "<tool_call>" not in none_prompt
    assert "offered as native tool calls" not in none_prompt
    assert "No Hermes tools are available" in none_prompt


def test_tool_call_parser_accepts_only_offered_valid_calls():
    valid = '<tool_call>{"id":"call_7","type":"function","function":{"name":"read_file","arguments":"{\\"path\\":\\"a.txt\\"}"}}</tool_call>'
    calls, content = extract_tool_calls("before\n" + valid + "\nafter", {"read_file"})
    assert len(calls) == 1
    assert calls[0].id == "call_7"
    assert calls[0].function.name == "read_file"
    assert calls[0].function.arguments == '{"path":"a.txt"}'
    assert content == "before\n\nafter"

    rejected = valid.replace("read_file", "shell")
    assert extract_tool_calls(rejected, {"read_file"}) == ([], rejected)
    malformed = valid.replace('{\\"path\\":\\"a.txt\\"}', "not-json")
    assert extract_tool_calls(malformed, {"read_file"}) == ([], malformed)


def test_images_are_bounded_and_reject_unsafe_urls():
    encoded = base64.b64encode(b"image").decode()
    content = [
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
        {"type": "image_url", "image_url": {"url": "http://example.com/no.png"}},
        {"type": "image_url", "image_url": {"url": "file:///etc/passwd"}},
    ]
    assert extract_images([{"role": "user", "content": content}]) == [
        {"url": "https://example.com/a.png"},
        {"data": encoded, "mimeType": "image/png"},
    ]


def test_sync_completion_uses_official_bridge_with_no_cursor_tools():
    client = CursorSDKClient(api_key="crsr_test", base_url=CURSOR_API_URL)
    try:
        response = client.chat.completions.create(
            model="cursor/composer-2.5",
            messages=[{"role": "user", "content": "hello"}],
            tools=[_tool()],
            timeout=15,
        )
        sdk = FakeCursorClient.instances[0]
        options = sdk.options[0]
        assert options["api_key"] == "crsr_test"
        assert options["model"] == "composer-2.5"
        assert options["tools"] == ["mcp"]
        assert options["mcp_servers"] == {}
        assert "sandbox_options" not in options["local"]
        assert not Path(options["local"]["cwd"]).exists()
        assert sdk.messages[0]["text"].endswith("Continue from the final transcript entry.")
        assert response.choices[0].message.content == "hello"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.prompt_tokens == 4
        assert response.usage.total_tokens == 6
        assert response.usage.prompt_tokens_details.cached_tokens == 1
        assert response.usage.prompt_tokens_details.cache_write_tokens == 2
        assert sdk.timeout_options["timeout"] == 15
    finally:
        client.close()
    assert sdk.agents[0].closed
    assert not sdk.closed


def test_completion_uses_streamed_text_when_wait_result_is_empty():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = ""
    sdk.result.stream_text = "streamed answer"
    try:
        response = client.chat.completions.create(
            model="auto", messages=[{"role": "user", "content": "hello"}]
        )
    finally:
        client.close()
    assert response.choices[0].message.content == "streamed answer"
    assert response.usage.total_tokens == 6


def test_terminal_sdk_error_is_not_misreported_as_empty_success():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = ""
    sdk.result.status = "ERROR"
    sdk.result.status_message = "You're out of usage. Switch to Auto."
    try:
        with pytest.raises(RuntimeError, match="out of usage") as exc_info:
            client.chat.completions.create(
                model="composer-2.5", messages=[{"role": "user", "content": "hello"}]
            )
    finally:
        client.close()
    assert exc_info.value.status_code == 429


def test_tool_call_and_sync_stream_are_openai_shaped():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = '<tool_call>{"function":{"name":"read_file","arguments":{"path":"x"}}}</tool_call>'
    try:
        chunks = list(client.chat.completions.create(
            model="auto", messages=[], tools=[_tool()], stream=True
        ))
    finally:
        client.close()
    calls = _stream_calls(chunks)
    assert calls[0].function.name == "read_file"
    finish = [
        chunk.choices[0].finish_reason
        for chunk in chunks
        if chunk.choices and chunk.choices[0].finish_reason
    ]
    assert finish == ["tool_calls"]
    assert sdk.options[0]["model"] == "auto"
    assert chunks[-1].choices == []
    assert chunks[-1].usage.total_tokens == 6


def test_async_completion_and_stream():
    client = CursorSDKClient(api_key="crsr_test")

    async def scenario():
        response = await client.chat.completions.create(model="auto", messages=[])
        stream = await client.chat.completions.create(model="auto", messages=[], stream=True)
        chunks = [chunk async for chunk in stream]
        return response, chunks

    try:
        response, chunks = asyncio.run(scenario())
    finally:
        client.close()
    assert response.choices[0].message.content == "hello"
    assert chunks[0].choices[0].delta.content == "hello"


def test_model_listing_uses_sdk_and_closes_bridge():
    assert list_cursor_models(api_key="crsr_test", timeout=4) == ["composer-2.5", "gpt-5.4"]
    sdk = FakeCursorClient.instances[0]
    assert sdk.list_api_key == "crsr_test"
    assert sdk.closed
    assert FakeCursorClient.launches[0]["allow_api_key_env_fallback"] is False


def test_missing_key_and_foreign_base_url_fail_before_launch():
    foreign = CursorSDKClient(api_key="secret", base_url="https://example.com")
    with pytest.raises(ValueError):
        foreign.chat.completions.create(model="auto", messages=[])
    foreign.close()
    client = CursorSDKClient(api_key="")
    try:
        with pytest.raises(RuntimeError, match="CURSOR_API_KEY"):
            client.chat.completions.create(model="auto", messages=[])
    finally:
        client.close()
    assert FakeCursorClient.launches == []


def test_session_reuses_agent_across_facades_and_sends_only_delta():
    first = CursorSDKClient(api_key="crsr_test")
    first.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "first-user"}],
        _cursor_session_scope="scope-1",
    )
    first.close()

    second = CursorSDKClient(api_key="crsr_test")
    second.chat.completions.create(
        model="auto",
        messages=[
            {"role": "user", "content": "first-user"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "second-user"},
        ],
        _cursor_session_scope="scope-1",
    )
    second.close()

    sdk = FakeCursorClient.instances[0]
    assert len(FakeCursorClient.instances) == 1
    assert len(sdk.agents) == 1
    assert len(sdk.agents[0].messages) == 2
    assert '"content":"second-user"' in sdk.agents[0].messages[1]["text"]
    assert '"content":"first-user"' not in sdk.agents[0].messages[1]["text"]
    assert not sdk.agents[0].closed


def test_session_tool_continuation_reuses_agent_without_replaying_history():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = '<tool_call>{"id":"call_1","function":{"name":"read_file","arguments":{"path":"x"}}}</tool_call>'
    first = client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "inspect-x"}],
        tools=[_tool()],
        _cursor_session_scope="scope-tool",
    )
    call = first.choices[0].message.tool_calls[0]
    sdk.result.result = "done"
    client.chat.completions.create(
        model="auto",
        messages=[
            {"role": "user", "content": "inspect-x"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": call.id, "content": "file-data"},
        ],
        tools=[_tool()],
        _cursor_session_scope="scope-tool",
    )
    client.close()

    assert len(sdk.agents) == 1
    assert len(sdk.agents[0].messages) == 2
    assert '"content":"file-data"' in sdk.agents[0].messages[1]["text"]
    assert "inspect-x" not in sdk.agents[0].messages[1]["text"]


def test_streamed_turn_updates_session_and_continuation_sends_only_delta():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = "first-answer"
    stream = client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "first-user"}],
        stream=True,
        _cursor_session_scope="scope-stream-continue",
    )
    chunks = list(stream)
    assert _stream_text(chunks) == "first-answer"

    sdk.result.result = "second-answer"
    client.chat.completions.create(
        model="auto",
        messages=[
            {"role": "user", "content": "first-user"},
            {"role": "assistant", "content": "first-answer"},
            {"role": "user", "content": "second-user"},
        ],
        _cursor_session_scope="scope-stream-continue",
    )
    client.close()

    sdk = FakeCursorClient.instances[0]
    assert len(sdk.agents) == 1
    assert len(sdk.agents[0].messages) == 2
    assert '"content":"second-user"' in sdk.agents[0].messages[1]["text"]
    assert '"content":"first-user"' not in sdk.agents[0].messages[1]["text"]


def test_session_isolates_scope_account_and_model():
    cases = [
        ("crsr_a", "auto", "scope-1"),
        ("crsr_a", "auto", "scope-2"),
        ("crsr_b", "auto", "scope-1"),
        ("crsr_a", "composer-2.5", "scope-1"),
    ]
    for api_key, model, scope in cases:
        client = CursorSDKClient(api_key=api_key)
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"{api_key}-{model}-{scope}"}],
            _cursor_session_scope=scope,
        )
        client.close()

    sdk = FakeCursorClient.instances[0]
    assert len(FakeCursorClient.instances) == 1
    assert len(sdk.agents) == 4
    assert len(client_module._SESSIONS) == 4


def test_history_rewrite_replaces_agent_and_cancel_preserves_session():
    client = CursorSDKClient(api_key="crsr_test")
    client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "original"}],
        _cursor_session_scope="scope-rewrite",
    )
    sdk = FakeCursorClient.instances[0]
    synthetic = FakeRun(sdk.result)
    client._runs.add(synthetic)
    client.cancel()
    client._runs.clear()
    assert synthetic.cancelled

    client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "rewritten"}],
        _cursor_session_scope="scope-rewrite",
    )
    client.close()

    assert len(sdk.agents) == 2
    assert sdk.agents[0].closed
    assert not sdk.agents[1].closed


def test_lru_evicts_closed_agent_without_deleting_bridge(monkeypatch):
    monkeypatch.setenv("HERMES_CURSOR_MAX_SESSIONS", "1")
    client = CursorSDKClient(api_key="crsr_test")
    for scope in ("scope-old", "scope-new"):
        client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": scope}],
            _cursor_session_scope=scope,
        )
    client.close()

    sdk = FakeCursorClient.instances[0]
    assert len(client_module._SESSIONS) == 1
    assert sdk.agents[0].closed
    assert not sdk.agents[1].closed
    assert not sdk.closed


def test_busy_session_uses_closed_one_shot_agent():
    client = CursorSDKClient(api_key="crsr_test")
    client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "first"}],
        _cursor_session_scope="scope-busy",
    )
    sdk = FakeCursorClient.instances[0]
    session = next(iter(client_module._SESSIONS.values()))
    session.lock.acquire()
    try:
        client.chat.completions.create(
            model="auto",
            messages=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "concurrent"},
            ],
            _cursor_session_scope="scope-busy",
        )
    finally:
        session.lock.release()
        client.close()

    assert len(sdk.agents) == 2
    assert not sdk.agents[0].closed
    assert sdk.agents[1].closed


def test_eviction_cleanup_cannot_delete_replacement_workspace(monkeypatch):
    monkeypatch.setenv("HERMES_CURSOR_MAX_SESSIONS", "1")
    home = client_module._home_key()
    key_old = (home, "scope-old", "account", "auto")
    key_new = (home, "scope-new", "account", "auto")
    old = client_module._lease_session(key_old)
    assert old is not None
    old_cwd = old.cwd
    client_module._release_session(old)

    cleanup_started = threading.Event()
    allow_cleanup = threading.Event()
    original_close = client_module._close_session

    def paused_close(session):
        if session is old:
            cleanup_started.set()
            assert allow_cleanup.wait(2)
        original_close(session)

    monkeypatch.setattr(client_module, "_close_session", paused_close)
    worker_result = []

    def evict_old():
        leased = client_module._lease_session(key_new)
        worker_result.append(leased)
        if leased is not None:
            client_module._release_session(leased)

    worker = threading.Thread(target=evict_old)
    worker.start()
    assert cleanup_started.wait(2)
    replacement = client_module._lease_session(key_old)
    assert replacement is not None
    try:
        assert replacement.cwd != old_cwd
        allow_cleanup.set()
        worker.join(2)
        assert not worker.is_alive()
        assert replacement.cwd.exists()
    finally:
        allow_cleanup.set()
        client_module._release_session(replacement)
    assert worker_result and worker_result[0] is not None


def test_failed_wait_cancels_in_flight_run():
    client = CursorSDKClient(api_key="crsr_test")
    agent = FakeAgent(FakeCursorClient())
    failed = FailingRun(agent.owner.result)
    agent.send = lambda message: failed
    with pytest.raises(TimeoutError, match="run timed out"):
        list(client._run_agent(agent, prompt="hello", images=[], allowed_names=set()))
    assert failed.cancelled
    assert failed not in client._runs


def test_stream_yields_first_chunk_before_run_completes():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = "hello"
    gate = threading.Event()

    def finish_text(run):
        # The SDK run is still in flight here; the test only opens the gate
        # after it has received the first chunk.
        assert gate.wait(10)
        return _assistant_event("lo")

    run = _scripted_run(sdk, [_assistant_event("hel"), finish_text])
    stream = client.chat.completions.create(model="auto", messages=[], stream=True)
    iterator = iter(stream)
    holder = {}

    def pull_first():
        holder["chunk"] = next(iterator, None)

    worker = threading.Thread(target=pull_first, daemon=True)
    worker.start()
    try:
        worker.join(2)
        assert not worker.is_alive(), "no chunk arrived before the SDK run completed"
        assert holder["chunk"].choices[0].delta.content == "hel"
        gate.set()
        rest = list(iterator)
    finally:
        gate.set()
        client.close()
        worker.join(5)
    assert _stream_text([holder["chunk"]]) + _stream_text(rest) == "hello"


def test_stream_splits_tool_block_at_arbitrary_boundaries():
    text = (
        'intro <tool_call>{"id":"c1","type":"function","function":{"name":"read_file",'
        '"arguments":"{\\"path\\":\\"a.txt\\"}"}}</tool_call> outro'
    )
    expected_calls, expected_content = extract_tool_calls(text, {"read_file"})
    assert expected_calls and expected_content == "intro  outro"
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    try:
        for split in range(len(text) + 1):
            _, calls, content = _collect_scripted(client, sdk, text, split)
            assert [call.function.name for call in calls] == ["read_file"], split
            assert calls[0].function.arguments == expected_calls[0].function.arguments, split
            assert content == expected_content, (split, content)
    finally:
        client.close()


@pytest.mark.parametrize(
    "block",
    [
        '<tool_call>{"function":{"name":"shell","arguments":"{}"}}</tool_call>',
        '<tool_call>{"function":{"name":"read_file","arguments":"not-json"}}</tool_call>',
    ],
)
def test_stream_keeps_unoffered_and_malformed_blocks_as_text(block):
    text = f"before {block} after"
    expected_calls, expected_content = extract_tool_calls(text, {"read_file"})
    assert expected_calls == [] and "tool_call" in expected_content
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    try:
        for split in (len("before <tool"), len(text) // 2, len(text) - 1):
            _, calls, content = _collect_scripted(client, sdk, text, split)
            assert calls == [], split
            assert content == expected_content, (split, content)
    finally:
        client.close()


def test_stream_does_not_repeat_terminal_text_already_streamed():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = "hello world"  # run.wait() repeats the streamed concatenation
    _scripted_run(sdk, [_assistant_event("hello "), _assistant_event("world")])
    try:
        chunks = list(client.chat.completions.create(model="auto", messages=[], stream=True))
    finally:
        client.close()
    assert _stream_text(chunks) == "hello world"


def test_stream_falls_back_to_terminal_result_when_no_fragments_arrive():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = "terminal only"
    sdk.result.stream_text = ""
    try:
        chunks = list(client.chat.completions.create(model="auto", messages=[], stream=True))
    finally:
        client.close()
    assert _stream_text(chunks) == "terminal only"
    assert chunks[-1].usage.total_tokens == 6


def test_stream_late_error_raises_and_clears_session():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = ""
    sdk.result.status_message = "You're out of usage. Switch to Auto."
    run = _scripted_run(
        sdk,
        [
            _assistant_event("partial"),
            _status_event("ERROR", "You're out of usage. Switch to Auto."),
        ],
    )
    stream = client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        _cursor_session_scope="scope-late-error",
    )
    seen = []
    try:
        with pytest.raises(client_module.CursorSDKRunError, match="out of usage") as exc_info:
            for chunk in stream:
                seen.append(chunk)
    finally:
        client.close()
    assert exc_info.value.status_code == 429
    assert _stream_text(seen) == "partial"
    assert not any(chunk.choices and chunk.choices[0].finish_reason for chunk in seen)
    session = next(iter(client_module._SESSIONS.values()))
    assert session.agent is None and session.history == []
    assert not session.lock.locked()
    assert run not in client._runs


def test_close_after_first_delta_cancels_in_flight_run():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = ""
    blocked = threading.Event()

    def block_until_cancelled(run):
        blocked.set()
        run.cancel_event.wait(10)
        return _status_event("CANCELLED", "offline cancelled run")

    run = _scripted_run(sdk, [_assistant_event("first"), block_until_cancelled])
    stream = client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        _cursor_session_scope="scope-close",
    )
    iterator = iter(stream)
    first = next(iterator)
    assert first.choices[0].delta.content == "first"
    drained = []

    def pull_rest():
        try:
            drained.append(list(iterator))
        except BaseException as exc:  # noqa: BLE001 - recorded for assertions
            drained.append(exc)

    worker = threading.Thread(target=pull_rest, daemon=True)
    worker.start()
    try:
        assert blocked.wait(5), "worker never reached the blocked stream read"
        stream.close()
        worker.join(5)
        assert not worker.is_alive()
    finally:
        stream.close()
        client.close()
        worker.join(5)
    assert run.cancelled
    assert isinstance(drained[0], client_module.CursorSDKRunError)
    assert list(stream) == []  # a closed stream never starts a second run
    session = next(iter(client_module._SESSIONS.values()))
    assert session.agent is None and session.history == []
    assert not session.lock.locked()


def test_async_stream_cancellation_cancels_run_and_releases_session():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = ""
    blocked = threading.Event()

    def block_until_cancelled(run):
        blocked.set()
        run.cancel_event.wait(10)
        return _status_event("CANCELLED", "offline cancelled run")

    run = _scripted_run(sdk, [_assistant_event("first"), block_until_cancelled])

    async def scenario():
        stream = client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            _cursor_session_scope="scope-async-cancel",
        )
        stream = await stream
        seen = []

        async def consume():
            async for chunk in stream:
                seen.append(chunk)

        task = asyncio.create_task(consume())
        assert await asyncio.to_thread(blocked.wait, 5), "async pull never reached the blocked read"
        # The SDK loop now runs ahead of the consumer (its work is pumped on a worker
        # thread), so "the SDK reached the next read" no longer implies the first
        # chunk was pulled. Wait for the delivery itself before cancelling.
        deadline = time.monotonic() + 5
        while not seen and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert seen, "the first chunk was never delivered"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return seen

    try:
        seen = asyncio.run(scenario())
    finally:
        client.close()
    assert seen and seen[0].choices[0].delta.content == "first"
    assert run.cancelled
    session = next(iter(client_module._SESSIONS.values()))
    deadline = time.monotonic() + 5
    while (session.agent is not None or session.lock.locked()) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert session.agent is None and session.history == []
    assert not session.lock.locked()


def test_stream_emits_plain_text_before_an_incomplete_tool_block_completes():
    """A tool-enabled turn holds its text until the decision point, then emits it whole."""
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    text = (
        'answer <tool_call>{"id":"c1","function":{"name":"read_file",'
        '"arguments":"{}"}}</tool_call> done'
    )
    sdk.result.result = text
    split = text.index("<tool_call>") + len("<tool_call>") + 3
    gate = threading.Event()

    def finish_block(run):
        assert gate.wait(10)  # the tool block is still incomplete here
        return _assistant_event(text[split:])

    _scripted_run(sdk, [_assistant_event(text[:split]), finish_block])
    stream = client.chat.completions.create(
        model="auto", messages=[], tools=[_tool()], stream=True
    )
    iterator = iter(stream)
    holder = {}

    def pull_first():
        holder["chunk"] = next(iterator, None)

    worker = threading.Thread(target=pull_first, daemon=True)
    worker.start()
    try:
        worker.join(2)
        assert worker.is_alive(), "tool-enabled text was emitted before the turn's decision point"
        gate.set()
        worker.join(5)
        assert not worker.is_alive()
        rest = list(iterator)
    finally:
        gate.set()
        client.close()
        worker.join(5)
    assert _stream_text([holder["chunk"]]) + _stream_text(rest) == "answer  done"
    calls = _stream_calls(rest)
    assert calls and calls[0].function.name == "read_file"


def test_async_stream_aclose_after_first_chunk_cancels_run():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = ""
    blocked = threading.Event()

    def block_until_cancelled(run):
        blocked.set()
        run.cancel_event.wait(10)
        return _status_event("CANCELLED", "offline cancelled run")

    run = _scripted_run(sdk, [_assistant_event("first"), block_until_cancelled])

    async def scenario():
        stream = await client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            _cursor_session_scope="scope-aclose",
        )
        first = await stream.__aiter__().__anext__()
        await stream.aclose()
        return first

    try:
        first = asyncio.run(scenario())
    finally:
        client.close()
    assert first.choices[0].delta.content == "first"
    assert run.cancelled
    session = next(iter(client_module._SESSIONS.values()))
    deadline = time.monotonic() + 5
    while (session.agent is not None or session.lock.locked()) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert session.agent is None and session.history == []
    assert not session.lock.locked()


def test_stream_tool_argument_can_contain_closing_tag():
    import json
    arguments = {"path": '</tool_call> and "quoted" \\ text'}
    text = '<tool_call>' + json.dumps({"function": {"name": "read_file", "arguments": arguments}}) + '</tool_call>'
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    try:
        for split in range(len(text) + 1):
            _, calls, content = _collect_scripted(client, sdk, text, split)
            assert len(calls) == 1, (split, content)
            assert json.loads(calls[0].function.arguments) == arguments
            assert not content
    finally:
        client.close()


@pytest.mark.parametrize("close_target", ["stream", "client"])
def test_close_during_send_cancels_new_run_and_releases_session(close_target):
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    started, release = threading.Event(), threading.Event()
    run = ScriptedRun(sdk.result, [_assistant_event("late")])
    agent = FakeAgent(sdk)

    def send(message):
        started.set()
        assert release.wait(5)
        return run

    agent.send = send
    sdk.create_agent = lambda options: agent
    stream = client.chat.completions.create(model="auto", messages=[], stream=True, _cursor_session_scope="closing-send")
    errors = []

    def pull():
        try:
            next(iter(stream), None)
        except client_module.CursorSDKRunError as exc:
            errors.append(exc)

    worker = threading.Thread(target=pull)
    worker.start()
    try:
        assert started.wait(2)
        (stream if close_target == "stream" else client).close()
        release.set()
        worker.join(3)
        assert not worker.is_alive()
        assert run.cancelled
        session = next(iter(client_module._SESSIONS.values()))
        assert session.agent is None and not session.lock.locked()
    finally:
        release.set()
        worker.join(5)
        stream.close()
        client.close()


def test_await_stream_resolves_to_async_iterable_without_starting_run():
    client = CursorSDKClient(api_key="crsr_test")

    async def scenario():
        stream = client.chat.completions.create(model="auto", messages=[], stream=True)
        resolved = await stream
        assert resolved is stream
        assert FakeCursorClient.launches == []  # awaiting must not collect the run
        return [chunk async for chunk in resolved]

    try:
        chunks = asyncio.run(scenario())
    finally:
        client.close()
    assert chunks[0].choices[0].delta.content == "hello"
    assert chunks[-1].usage.total_tokens == 6


@pytest.mark.parametrize("mode", ["relay", "await", "async_for"])
def test_lazy_stream_works_inside_event_loop_without_blocking(monkeypatch, mode):
    client = CursorSDKClient(api_key="crsr_test")
    original_wait = FakeRun.wait

    def checked_wait(run):
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        return original_wait(run)

    monkeypatch.setattr(FakeRun, "wait", checked_wait)

    async def scenario():
        stream = client.chat.completions.create(model="auto", messages=[], stream=True)
        assert FakeCursorClient.launches == []
        if mode == "relay":
            iterator = iter(stream)  # Relay constructs/iterates on-loop, pulls off-loop.
            assert FakeCursorClient.launches == []
            chunks = await asyncio.to_thread(list, iterator)
            assert list(stream) == []  # A consumed stream must not start a second run.
        elif mode == "await":
            chunks = [chunk async for chunk in await stream]
        else:
            chunks = [chunk async for chunk in stream]
        stream.close()
        return chunks

    try:
        chunks = asyncio.run(scenario())
    finally:
        client.close()
    assert chunks[0].choices[0].delta.content == "hello"
    assert chunks[-1].usage.total_tokens == 6
    assert len(FakeCursorClient.instances[0].messages) == 1


def test_close_unconsumed_stream_never_launches_sdk():
    client = CursorSDKClient(api_key="crsr_test")
    stream = client.chat.completions.create(model="auto", messages=[], stream=True)
    stream.close()
    assert list(stream) == []
    client.close()
    assert FakeCursorClient.launches == []


def test_lazy_stream_preserves_provider_error_and_client_cancellation():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    started, cancelled = threading.Event(), threading.Event()

    class CancellableRun(FakeRun):
        def stream(self):
            started.set()
            assert cancelled.wait(5), "Client cancellation did not reach the SDK run"
            yield _status_event("CANCELLED", "offline cancelled run")

        def cancel(self):
            super().cancel()
            cancelled.set()

    run = CancellableRun(sdk.result)
    agent = FakeAgent(sdk)
    agent.send = lambda message: run
    sdk.create_agent = lambda options: agent

    async def scenario():
        stream = client.chat.completions.create(model="auto", messages=[], stream=True)
        task = asyncio.create_task(asyncio.to_thread(list, iter(stream)))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            client.close()
            with pytest.raises(client_module.CursorSDKRunError, match="offline cancelled run"):
                await task
        finally:
            cancelled.set()
            await asyncio.gather(task, return_exceptions=True)
            stream.close()

    asyncio.run(scenario())
    assert run.cancelled and not client._runs
    assert agent.closed


# --- model parameter selection (model@key=value,...) ---

from hermes_cursor_provider.client import _model_id, _model_selection


@pytest.mark.parametrize(
    "name, expected",
    [
        (None, "auto"),
        ("", "auto"),
        ("default", "auto"),
        ("grok-4.7", "grok-4.7"),
        ("cursor/grok-4.7", "grok-4.7"),
        (
            "grok-4.7@reasoning_effort=xhigh,fast=true",
            {"id": "grok-4.7", "params": [
                {"id": "reasoning_effort", "value": "xhigh"},
                {"id": "fast", "value": "true"},
            ]},
        ),
        (
            "grok-4.7@context=500k,reasoning_effort=xhigh,fast=true",
            {"id": "grok-4.7", "params": [
                {"id": "context", "value": "500k"},
                {"id": "reasoning_effort", "value": "xhigh"},
                {"id": "fast", "value": "true"},
            ]},
        ),
        (
            "cursor:claude-opus-5-5@ context=1m , effort=max ",
            {"id": "claude-opus-5-5", "params": [
                {"id": "context", "value": "1m"},
                {"id": "effort", "value": "max"},
            ]},
        ),
    ],
)
def test_model_selection_parses_params(name, expected):
    assert _model_selection(name) == expected


@pytest.mark.parametrize(
    "name",
    ["grok-4.7@", "grok-4.7@xhigh", "grok-4.7@=x", "grok-4.7@fast=", "@fast=true",
     "grok-4.7@fast=true,fast=false"],
)
def test_model_selection_rejects_malformed_params(name):
    with pytest.raises(ValueError, match="Cursor model parameters"):
        _model_selection(name)


def test_create_agent_sends_parameterised_selection():
    client = CursorSDKClient(api_key="crsr_test")
    try:
        client.chat.completions.create(
            model="grok-4.7@reasoning_effort=xhigh,fast=true",
            messages=[{"role": "user", "content": "hello"}],
        )
    finally:
        client.close()
    sdk = FakeCursorClient.instances[0]
    assert sdk.options[0]["model"] == {"id": "grok-4.7", "params": [
        {"id": "reasoning_effort", "value": "xhigh"},
        {"id": "fast", "value": "true"},
    ]}


def test_malformed_params_fail_before_bridge_launch():
    client = CursorSDKClient(api_key="crsr_test")
    try:
        with pytest.raises(ValueError, match="Cursor model parameters"):
            client.chat.completions.create(
                model="grok-4.7@xhigh", messages=[{"role": "user", "content": "hi"}]
            )
    finally:
        client.close()
    assert FakeCursorClient.launches == []


def test_sessions_isolate_parameter_variants():
    for model in ("grok-4.7", "grok-4.7@reasoning_effort=xhigh", "grok-4.7@reasoning_effort=low"):
        client = CursorSDKClient(api_key="crsr_a")
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": model}],
            _cursor_session_scope="scope-1",
        )
        client.close()
    assert len(FakeCursorClient.instances[0].agents) == 3
    assert len(client_module._SESSIONS) == 3


def test_model_id_keeps_params_in_normalised_name():
    assert _model_id("cursor/grok-4.7@ fast=true ") == "grok-4.7@fast=true"


# --- opt-in parameter variants in the model listing (HERMES_CURSOR_MODEL_VARIANTS) ---


def test_listing_without_variants_env_is_unchanged(monkeypatch):
    monkeypatch.delenv("HERMES_CURSOR_MODEL_VARIANTS", raising=False)
    assert list_cursor_models(api_key="crsr_test") == ["composer-2.5", "gpt-5.4"]


def test_listing_appends_configured_variants_in_normalised_form(monkeypatch):
    monkeypatch.setenv(
        "HERMES_CURSOR_MODEL_VARIANTS",
        " composer-2.5@fast=false ; cursor/gpt-5.4@ reasoning_effort=high , fast=true ;",
    )
    assert list_cursor_models(api_key="crsr_test") == [
        "composer-2.5",
        "gpt-5.4",
        "composer-2.5@fast=false",
        "gpt-5.4@reasoning_effort=high,fast=true",
    ]


def test_listing_skips_malformed_unknown_bare_and_duplicate_variants(monkeypatch):
    monkeypatch.setenv(
        "HERMES_CURSOR_MODEL_VARIANTS",
        "composer-2.5@xhigh;not-a-model@fast=true;composer-2.5;"
        "composer-2.5@fast=true;composer-2.5@fast=true",
    )
    assert list_cursor_models(api_key="crsr_test") == [
        "composer-2.5",
        "gpt-5.4",
        "composer-2.5@fast=true",
    ]


# --- plan-only replies: nudge once for the <tool_call> before ending the turn ---

_CALL = '<tool_call>{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\\"path\\":\\"FIX.md\\"}"}}</tool_call>'


def _sequenced(sdk, texts):
    """Agent whose successive send() calls return the given reply texts in order."""
    agent = FakeAgent(sdk)
    replies = list(texts)

    def send(message):
        agent.messages.append(message)
        text = replies.pop(0) if replies else ""
        return FakeRun(FakeResult(result=text, model=sdk.result.model, usage=sdk.result.usage))

    agent.send = send
    sdk.create_agent = lambda options: agent
    return agent


def _create(client, tools=True, stream=False):
    return client.chat.completions.create(
        model="grok-4.7",
        messages=[{"role": "user", "content": "apply the fix note"}],
        tools=[_tool()] if tools else None,
        stream=stream,
    )


def _sdk_client():
    client = CursorSDKClient(api_key="crsr_test")
    _create(client)  # launch the bridge once so the fake SDK instance exists
    return client, FakeCursorClient.instances[-1]


@pytest.mark.parametrize("plan", [
    "I'll start by reading the fix note so the groups are applied exactly.",
    "Let me read the fix instructions first.",
    "I will read FIX.md and then apply group 1:",
    "First, I'll read the review file.",
])
def test_plan_only_reply_is_nudged_once_into_a_tool_call(plan):
    client, sdk = _sdk_client()
    agent = _sequenced(sdk, [plan, _CALL])
    response = _create(client)
    assert len(agent.messages) == 2
    assert "call it now using the tools provided in this conversation" in agent.messages[1]["text"]
    assert "FINAL:" in agent.messages[1]["text"]
    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.tool_calls[0].function.name == "read_file"


def test_nudge_is_attempted_only_once():
    client, sdk = _sdk_client()
    agent = _sequenced(sdk, ["I'll read the file.", "I'll read the file now.", _CALL])
    with pytest.raises(client_module.CursorSDKRunError) as exc_info:
        _create(client)
    assert len(agent.messages) == 2
    assert exc_info.value.error_code == "cursor_plan_only"


@pytest.mark.parametrize("final", ["alpha-beta", "All 4 groups are committed and pushed.", "Done: 391"])
def test_real_final_answers_are_not_nudged(final):
    client, sdk = _sdk_client()
    agent = _sequenced(sdk, [final])
    response = _create(client)
    assert len(agent.messages) == 1
    assert response.choices[0].message.content == final


def test_plan_only_reply_without_tools_is_not_nudged():
    client, sdk = _sdk_client()
    agent = _sequenced(sdk, ["I'll explain the plan."])
    _create(client, tools=False)
    assert len(agent.messages) == 1


def test_nudge_can_be_disabled(monkeypatch):
    monkeypatch.setenv("HERMES_CURSOR_PLAN_NUDGE", "0")
    client, sdk = _sdk_client()
    agent = _sequenced(sdk, ["I'll read the file.", _CALL])
    _create(client)
    assert len(agent.messages) == 1


def test_streamed_plan_only_reply_is_nudged_into_a_tool_call():
    client, sdk = _sdk_client()
    agent = _sequenced(sdk, ["I'll read the fix note first.", _CALL])
    chunks = list(_create(client, stream=True))
    assert len(agent.messages) == 2
    calls = _stream_calls(chunks)
    assert calls and calls[0].function.name == "read_file"
    finals = [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason]
    assert finals == ["tool_calls"]


@pytest.mark.parametrize(
    "reply",
    ["Let me know if you'd like more.", "Would you like me to continue?", "I'll be happy to help."],
)
def test_pleasantry_only_replies_are_not_nudged(reply):
    client, sdk = _sdk_client()
    agent = _sequenced(sdk, [reply, _CALL])
    response = _create(client)
    assert len(agent.messages) == 1
    assert response.choices[0].finish_reason == "stop"
    assert response.choices[0].message.content == reply


def test_mid_sentence_announcement_is_nudged_once():
    client, sdk = _sdk_client()
    agent = _sequenced(sdk, ["The plan is loaded. Next I'll check the groups.", _CALL])
    response = _create(client)
    assert len(agent.messages) == 2
    assert response.choices[0].finish_reason == "tool_calls"


def test_announcement_with_pleasantry_is_still_nudged():
    client, sdk = _sdk_client()
    agent = _sequenced(sdk, ["I'll read the file. Let me know if you'd like more.", _CALL])
    response = _create(client)
    assert len(agent.messages) == 2
    assert response.choices[0].finish_reason == "tool_calls"


def test_final_prefixed_answer_is_never_raised():
    """The marker, not the plan heuristic, accepts the nudged reply."""
    client, sdk = _sdk_client()
    agent = _sequenced(
        sdk, ["I'll check the groups.", "FINAL: I'll re-read it and then answer."]
    )
    response = _create(client)
    assert len(agent.messages) == 2
    assert response.choices[0].finish_reason == "stop"
    content = response.choices[0].message.content
    assert content == "I'll check the groups.\n\nI'll re-read it and then answer."
    assert "FINAL:" not in content


def test_final_marker_is_stripped_from_delivered_text():
    client, sdk = _sdk_client()
    _sequenced(sdk, ["I'll check the groups.", "FINAL: all four groups are green."])
    response = _create(client)
    content = response.choices[0].message.content
    assert content == "I'll check the groups.\n\nall four groups are green."
    assert "FINAL:" not in content


def test_final_marker_is_stripped_when_the_nudged_reply_also_calls():
    """The marker must not survive the join just because the reply also carries a call."""
    client, sdk = _sdk_client()
    _sequenced(
        sdk,
        ["I'll check the groups.", "FINAL: I read it and the answer is 42.\n\n" + _CALL],
    )
    response = _create(client)
    assert response.choices[0].finish_reason == "tool_calls"
    content = response.choices[0].message.content
    assert content == "I'll check the groups.\n\nI read it and the answer is 42."
    assert "FINAL:" not in content


def test_final_marker_is_not_stripped_mid_text():
    """The strip is a prefix rule: text that never passed the gate is returned unchanged.

    Both ``_strip_final_marker`` call sites pass text that was not prefix-checked, so an
    unanchored pattern would delete a mid-sentence ``FINAL:`` and the space before it.
    """
    text = "See the FINAL: stage now"
    assert client_module._strip_final_marker(text) == text
    assert client_module._strip_final_marker("  FINAL: done") == "done"
    assert client_module._strip_final_marker("\n\nfinal: done") == "done"


def test_final_marker_is_accepted_after_a_blank_line():
    client, sdk = _sdk_client()
    _sequenced(sdk, ["I'll check the groups.", "\n\nFINAL: all four groups are green."])
    response = _create(client)
    content = response.choices[0].message.content
    assert content.endswith("all four groups are green.")
    assert "FINAL:" not in content


def test_quoted_marker_is_not_a_final_answer():
    """The marker must PREFIX the reply; quoting the nudge does not pass."""
    client, sdk = _sdk_client()
    _sequenced(
        sdk,
        [
            "I'll check the groups.",
            "I could not finish. The nudge asked me to prefix the answer with FINAL: but I did not.",
        ],
    )
    with pytest.raises(client_module.CursorSDKRunError) as exc_info:
        _create(client)
    assert exc_info.value.error_code == "cursor_plan_only"


def test_legitimate_answer_with_ill_is_nudged_then_marked_final():
    """A false-positive nudge costs one follow-up, never a hard failure, when marked."""
    reply = "I'll note that the file is 391 lines long."
    client, sdk = _sdk_client()
    _sequenced(sdk, [reply, "FINAL: " + reply])
    response = _create(client)
    content = response.choices[0].message.content
    assert content == reply + "\n\n" + reply
    assert "FINAL:" not in content


# --- Hermes tools inside Cursor's native loop (custom_tools) ---


def test_offered_tools_are_custom_tools():
    client = CursorSDKClient(api_key="crsr_test")
    try:
        client.chat.completions.create(
            model="grok-4.7",
            messages=[{"role": "user", "content": "say hello"}],
            tools=[_tool()],
        )
    finally:
        client.close()
    sdk = FakeCursorClient.instances[0]
    local = sdk.options[0]["local"]
    assert set(local["custom_tools"]) == {"read_file"}
    assert local["custom_tools"]["read_file"]["description"] == (
        "Read one file" + client_module._TOOL_DESCRIPTION_SUFFIX
    )
    assert local["custom_tools"]["read_file"]["input_schema"] == _tool()["function"]["parameters"]
    assert callable(local["custom_tools"]["read_file"]["execute"])


def test_create_agent_offers_only_mcp_tools():
    client = CursorSDKClient(api_key="crsr_test")
    try:
        client.chat.completions.create(
            model="grok-4.7",
            messages=[{"role": "user", "content": "say hello"}],
            tools=[_tool()],
        )
    finally:
        client.close()
    options = FakeCursorClient.instances[0].options[0]
    # "mcp" is the only capability group that still offers custom_tools; it exposes
    # no Cursor built-ins (probe-variants.json: every event is name='mcp').
    assert options["tools"] == ["mcp"]
    assert options["mcp_servers"] == {}


def test_tool_choice_limits_registered_tools():
    client = CursorSDKClient(api_key="crsr_test")
    try:
        client.chat.completions.create(
            model="grok-4.7",
            messages=[{"role": "user", "content": "hello"}],
            tools=[_tool(), _tool("write_file")],
            tool_choice={"type": "function", "function": {"name": "read_file"}},
        )
        client.chat.completions.create(
            model="grok-4.7",
            messages=[{"role": "user", "content": "hello"}],
            tools=[_tool()],
            tool_choice="none",
        )
    finally:
        client.close()
    sdk = FakeCursorClient.instances[0]
    assert set(sdk.options[0]["local"]["custom_tools"]) == {"read_file"}
    assert sdk.options[0]["tools"] == ["mcp"]
    # tool_choice="none" offers nothing at all, so the text-only shape is kept.
    assert "custom_tools" not in sdk.options[1]["local"]
    assert sdk.options[1]["tools"] == []


@pytest.mark.parametrize("mode", ["one-shot", "session", "busy-session"])
def test_native_prompt_reaches_the_sdk_at_every_call_site(mode):
    """Pin the switch where ``_execute`` flips it, not only in ``render_prompt``'s own test.

    The prompt is rendered at three production call sites (the one-shot run, the session
    run, and the busy-session one-shot fallback). A ``native_tools=False`` at any of them
    would restore the text protocol while the direct ``render_prompt`` test stayed green.
    """
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    scope = "scope-native-prompt"
    try:
        client.chat.completions.create(
            model="grok-4.7",
            messages=[{"role": "user", "content": "read the file"}],
            tools=[_tool()],
            _cursor_session_scope=None if mode == "one-shot" else scope,
        )
        if mode == "busy-session":
            session = next(iter(client_module._SESSIONS.values()))
            assert session.lock.acquire(blocking=False)
            try:
                client.chat.completions.create(
                    model="grok-4.7",
                    messages=[
                        {"role": "user", "content": "read the file"},
                        {"role": "assistant", "content": "hello"},
                        {"role": "user", "content": "continue"},
                    ],
                    tools=[_tool()],
                    _cursor_session_scope=scope,
                )
            finally:
                session.lock.release()
    finally:
        client.close()
    # The last agent is the one this mode's call site served: the session agent for
    # one-shot/session, the closing one-shot agent for the busy-session fallback.
    sent = sdk.agents[-1].messages[-1]["text"]
    assert "offered as native tool calls" in sent
    assert "call it directly with the native tool-call mechanism" in sent
    assert "<tool_call>" not in sent
    assert '"name":"read_file"' in sent


def test_no_tools_turn_never_gets_the_native_prompt():
    """Without tools the SDK keeps the plain-text instruction; the native switch is off."""
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    try:
        client.chat.completions.create(
            model="grok-4.7", messages=[{"role": "user", "content": "hello"}]
        )
    finally:
        client.close()
    sent = sdk.agents[0].messages[0]["text"]
    assert "offered as native tool calls" not in sent
    assert "<tool_call>" not in sent
    assert "No Hermes tools are available for this request. Answer in plain text." in sent


def test_no_tools_keeps_text_protocol():
    """Regression guard: without Hermes tools Cursor still gets no tools at all."""
    client = CursorSDKClient(api_key="crsr_test")
    try:
        response = client.chat.completions.create(
            model="grok-4.7",
            messages=[{"role": "user", "content": "hello"}],
        )
    finally:
        client.close()
    options = FakeCursorClient.instances[0].options[0]
    assert options["tools"] == []
    assert "custom_tools" not in options["local"]
    assert response.choices[0].finish_reason == "stop"


def test_callback_returns_tool_call_to_hermes():
    client = CursorSDKClient(api_key="crsr_test")
    try:
        response = client.chat.completions.create(
            model="grok-4.7",
            messages=[{"role": "user", "content": "CALL hermes_ping with token probe-1"}],
            tools=[_tool("hermes_ping")],
        )
    finally:
        client.close()
    sdk = FakeCursorClient.instances[0]
    assert sdk.tool_callbacks == [client_module._ACK_ACCEPTED]
    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    call = choice.message.tool_calls[0]
    assert call.id == call.call_id == "call_probe"
    assert call.type == "function"
    assert call.function.name == "hermes_ping"
    assert call.function.arguments == '{"token":"probe-1"}'


def _native_turn_messages(content, call):
    return [
        {"role": "user", "content": "CALL hermes_ping first"},
        {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call.id, "content": "pong:probe-1"},
        {"role": "user", "content": "CALL hermes_ping again"},
    ]


def test_second_turn_call_lands_on_that_turn():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    try:
        first = client.chat.completions.create(
            model="grok-4.7",
            messages=[{"role": "user", "content": "CALL hermes_ping first"}],
            tools=[_tool("hermes_ping")],
            _cursor_session_scope="scope-native-turn2",
        )
        call = first.choices[0].message.tool_calls[0]
        second = client.chat.completions.create(
            model="grok-4.7",
            messages=_native_turn_messages(first.choices[0].message.content, call),
            tools=[_tool("hermes_ping")],
            _cursor_session_scope="scope-native-turn2",
        )
    finally:
        client.close()
    assert len(sdk.agents) == 1
    assert len(sdk.agents[0].messages) == 2
    assert len(first.choices[0].message.tool_calls) == 1
    assert len(second.choices[0].message.tool_calls) == 1
    assert second.choices[0].message.tool_calls[0].id == "call_probe"


def test_assistant_echo_round_trips_for_next_turn():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    try:
        first = client.chat.completions.create(
            model="grok-4.7",
            messages=[{"role": "user", "content": "CALL hermes_ping first"}],
            tools=[_tool("hermes_ping")],
            _cursor_session_scope="scope-native-echo",
        )
        call = first.choices[0].message.tool_calls[0]
        messages = _native_turn_messages(first.choices[0].message.content, call)
        messages[-1] = {"role": "user", "content": "thanks, continue"}
        second = client.chat.completions.create(
            model="grok-4.7",
            messages=messages,
            tools=[_tool("hermes_ping")],
            _cursor_session_scope="scope-native-echo",
        )
    finally:
        client.close()
    # The assistant echo round-trips, so the `extends` check keeps the same agent.
    assert len(sdk.agents) == 1
    assert len(sdk.agents[0].messages) == 2
    delta = sdk.agents[0].messages[1]["text"]
    assert '"content":"thanks, continue"' in delta
    assert '"tool_call_id":"call_probe"' in delta
    assert '"content":"CALL hermes_ping first"' not in delta
    assert second.choices[0].finish_reason == "stop"


def test_stale_callback_name_is_dropped():
    """A callback bound to an earlier turn's set is refused by the LIVE allowed set."""
    holder = {
        "sink": [],
        "allowed": {"read_file"},
        "registered": {"read_file"},
        "active": True,
    }
    specs = [{"name": "read_file", "description": "", "parameters": {"type": "object"}}]
    tools = client_module.hermes_tools_as_custom_tools(specs, holder)
    stale = tools["read_file"]["execute"]
    assert stale({"path": "x"}, SimpleNamespace(tool_call_id="call_stale")) == (
        client_module._ACK_ACCEPTED
    )
    assert [entry["tool_call_id"] for entry in holder["sink"]] == ["call_stale"]

    # Rebind the SAME dict for the next turn, which did not offer read_file.
    holder["sink"] = []
    holder["allowed"] = {"write_file"}
    assert stale({"path": "x"}, SimpleNamespace(tool_call_id="call_stale")) == (
        "refused: tool not offered this turn"
    )
    assert holder["sink"] == []

    # Outside the run window, and without an id, nothing is ever accepted.
    holder["active"] = False
    assert stale({"path": "x"}, SimpleNamespace(tool_call_id="call_x")) == "refused: no active turn"
    holder["allowed"] = {"read_file"}
    holder["active"] = True
    assert stale({"path": "x"}, SimpleNamespace(tool_call_id=None)).startswith("refused:")
    assert holder["sink"] == []


def test_turn_cap_refuses_the_sixty_fifth_callback():
    """The 64-call cap is a hard stop: the 65th callback is refused, not queued."""
    holder = {
        "sink": [],
        "allowed": {"read_file"},
        "registered": {"read_file"},
        "active": True,
    }
    specs = [{"name": "read_file", "description": "", "parameters": {"type": "object"}}]
    execute = client_module.hermes_tools_as_custom_tools(specs, holder)["read_file"]["execute"]
    acks = [
        execute({"path": f"f{index}.txt"}, SimpleNamespace(tool_call_id=f"call_{index}"))
        for index in range(65)
    ]
    assert acks[:64] == [client_module._ACK_ACCEPTED] * 64
    assert acks[64] == client_module._ACK_REFUSED_LIMIT.format(cap=64)
    assert len(holder["sink"]) == 64
    assert [entry["tool_call_id"] for entry in holder["sink"]] == [
        f"call_{index}" for index in range(64)
    ]


# --- honest ack, run-scoped dedupe, configurable cap ---

_ACK_ACCEPTED_TEXT = (
    "accepted by Hermes: this tool is queued and runs after this turn; its result "
    "arrives in the next message. Do not call it again with the same arguments."
)
_ACK_DUPLICATE_TEXT = (
    "already accepted this turn \u2014 this exact call is queued once; do not repeat it. "
    "Its result arrives in the next message."
)
_ACK_REFUSED_LIMIT_TEXT = (
    "refused: turn limit of {cap} tool calls reached; the calls already accepted will "
    "still return their results. State your final answer now."
)
_TOOL_DESCRIPTION_SUFFIX_TEXT = (
    "\n\n(Hermes executes this tool after your turn; its result arrives in the next "
    "message. Do not repeat a call you have already made this turn.)"
)


def _seeded(gen, ids):
    """A holder as `_run_once` leaves it: this run's generation and its reconciled ids."""
    return {
        "sink": [],
        "allowed": {"read_file"},
        "registered": {"read_file"},
        "active": True,
        "gen": gen,
        "event_ids": set(ids),
    }


def _plain():
    """The four pre-existing keys only: what a hand-built holder looked like before."""
    return {
        "sink": [],
        "allowed": {"read_file"},
        "registered": {"read_file"},
        "active": True,
    }


def _execute_on(holder, name="read_file"):
    specs = [{"name": name, "description": "", "parameters": {"type": "object"}}]
    return client_module.hermes_tools_as_custom_tools(specs, holder)[name]["execute"]


def _ctx(call_id):
    return SimpleNamespace(tool_call_id=call_id)


def _entry(gen, call_id, path="a.txt", name="read_file"):
    """A full pre-seeded sink entry: the predicate reads name, arguments, id and gen."""
    return {"name": name, "arguments": {"path": path}, "tool_call_id": call_id, "gen": gen}


def test_accepted_ack_is_the_honest_wording_and_appends_one_entry():
    """Test 1: the in-turn signal says accepted/queued, not a bare 'queued'."""
    assert client_module._ACK_ACCEPTED == _ACK_ACCEPTED_TEXT
    holder = _plain()
    ack = _execute_on(holder)({"path": "a.txt"}, _ctx("call_1"))
    assert ack == client_module._ACK_ACCEPTED
    assert ack != "queued"
    assert [entry["tool_call_id"] for entry in holder["sink"]] == ["call_1"]


def test_identical_repeat_in_one_run_is_collapsed():
    """Test 2: the same (name, args) twice in one run is queued once and acked as such."""
    assert client_module._ACK_DUPLICATE == _ACK_DUPLICATE_TEXT
    holder = _seeded(2, ["call_1"])
    execute = _execute_on(holder)
    assert execute({"path": "a.txt"}, _ctx("call_1")) == client_module._ACK_ACCEPTED
    assert execute({"path": "a.txt"}, _ctx("call_2")) == client_module._ACK_DUPLICATE
    assert len(holder["sink"]) == 1
    # Guard order is active -> allowed -> call-id -> dedupe: an id-less repeat still takes
    # the id refusal and is never answered with a positive ack.
    idless = execute({"path": "a.txt"}, _ctx(None))
    assert idless.startswith("refused:")
    assert idless != client_module._ACK_DUPLICATE
    assert len(holder["sink"]) == 1


def test_duplicate_does_not_consume_the_cap(monkeypatch):
    """Test 3: A, A(dup), B all pass; only a third distinct call is refused at cap 2.
    Also pins the guard order: at the cap, a repeat still collapses (dedupe before cap)."""
    monkeypatch.setenv("HERMES_CURSOR_MAX_TURN_TOOL_CALLS", "2")
    holder = _seeded(2, ["call_1"])
    execute = _execute_on(holder)
    assert execute({"path": "a.txt"}, _ctx("call_1")) == client_module._ACK_ACCEPTED
    assert execute({"path": "a.txt"}, _ctx("call_2")) == client_module._ACK_DUPLICATE
    assert execute({"path": "b.txt"}, _ctx("call_3")) == client_module._ACK_ACCEPTED
    assert execute({"path": "c.txt"}, _ctx("call_4")) == (
        client_module._ACK_REFUSED_LIMIT.format(cap=2)
    )
    # The guard order itself: with the sink already AT the cap, a repeat of a *reconciled*
    # accepted call must still take the duplicate ack. If the cap were checked before the
    # dedupe this would be the limit refusal, and nothing else in the suite would notice.
    # (Only `call_1` is in this holder's event_ids, so only a repeat of `a.txt` is eligible.)
    assert execute({"path": "a.txt"}, _ctx("call_5")) == client_module._ACK_DUPLICATE
    assert [entry["tool_call_id"] for entry in holder["sink"]] == ["call_1", "call_3"]


@pytest.mark.parametrize(
    ("raw", "expected", "warns"),
    [
        (None, 64, False),
        ("", 64, False),
        ("   ", 64, False),
        ("abc", 64, True),
        ("0", 64, True),
        ("-3", 64, True),
        ("5", 5, False),
        ("1", 1, False),
        ("1024", 1024, False),
        ("99999", 1024, True),
    ],
)
def test_max_turn_tool_calls_env_semantics(monkeypatch, caplog, raw, expected, warns):
    """Test 4: unset/blank -> default; malformed or <= 0 -> default + warning; else clamped."""
    if raw is None:
        monkeypatch.delenv("HERMES_CURSOR_MAX_TURN_TOOL_CALLS", raising=False)
    else:
        monkeypatch.setenv("HERMES_CURSOR_MAX_TURN_TOOL_CALLS", raw)
    with caplog.at_level(logging.WARNING, logger=client_module.__name__):
        value = client_module._max_turn_tool_calls()
    assert value == expected
    assert isinstance(value, int)
    warned = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert bool(warned) is warns
    assert client_module._DEFAULT_MAX_TURN_TOOL_CALLS == 64
    assert client_module._MAX_MAX_TURN_TOOL_CALLS == 1024


def test_limit_refusal_tells_the_model_what_still_happens(monkeypatch):
    """Test 5: the refusal says the limit was reached and accepted calls still return."""
    assert client_module._ACK_REFUSED_LIMIT == _ACK_REFUSED_LIMIT_TEXT
    monkeypatch.setenv("HERMES_CURSOR_MAX_TURN_TOOL_CALLS", "1")
    holder = _plain()
    execute = _execute_on(holder)
    assert execute({"path": "a.txt"}, _ctx("call_1")) == client_module._ACK_ACCEPTED
    refusal = execute({"path": "b.txt"}, _ctx("call_2"))
    assert "turn limit" in refusal
    assert "still return their results" in refusal
    assert refusal == _ACK_REFUSED_LIMIT_TEXT.format(cap=1)
    assert len(holder["sink"]) == 1


def test_description_suffix_is_on_every_custom_tool():
    """Test 6: each registered tool says Hermes runs it later; None/missing -> suffix alone."""
    assert client_module._TOOL_DESCRIPTION_SUFFIX == _TOOL_DESCRIPTION_SUFFIX_TEXT
    specs = [
        {"name": "alpha", "description": "Alpha tool", "parameters": {"type": "object"}},
        {"name": "beta", "description": None, "parameters": {"type": "object"}},
        {"name": "gamma", "parameters": {"type": "object"}},
    ]
    tools = client_module.hermes_tools_as_custom_tools(specs, {})
    assert tools["alpha"]["description"] == "Alpha tool" + _TOOL_DESCRIPTION_SUFFIX_TEXT
    assert tools["beta"]["description"] == _TOOL_DESCRIPTION_SUFFIX_TEXT
    assert tools["gamma"]["description"] == _TOOL_DESCRIPTION_SUFFIX_TEXT


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF", " 0 "])
def test_dedupe_kill_switch_lets_the_repeat_through(monkeypatch, value):
    """Test 7: HERMES_CURSOR_DEDUPE off -> the repeat is accepted; on, the same fixture dedupes."""
    control = _seeded(2, ["call_1"])
    execute = _execute_on(control)
    execute({"path": "a.txt"}, _ctx("call_1"))
    assert execute({"path": "a.txt"}, _ctx("call_2")) == client_module._ACK_DUPLICATE

    monkeypatch.setenv("HERMES_CURSOR_DEDUPE", value)
    holder = _seeded(2, ["call_1"])
    execute = _execute_on(holder)
    assert execute({"path": "a.txt"}, _ctx("call_1")) == client_module._ACK_ACCEPTED
    assert execute({"path": "a.txt"}, _ctx("call_2")) == client_module._ACK_ACCEPTED
    assert len(holder["sink"]) == 2


def test_older_generation_entry_does_not_shadow_a_repeat():
    """Test 8: an earlier run's entry (gen 1) never suppresses the current run's (gen 2) repeat."""
    holder = _seeded(2, ["call_1"])
    holder["sink"].append(_entry(1, "call_1"))
    ack = _execute_on(holder)({"path": "a.txt"}, _ctx("call_2"))
    assert ack == client_module._ACK_ACCEPTED
    assert [(e["tool_call_id"], e["gen"]) for e in holder["sink"]] == [("call_1", 1), ("call_2", 2)]


def test_unreconciled_entry_does_not_shadow_a_live_repeat():
    """Test 9: an entry whose id is not in this run's event ids cannot suppress a live call."""
    holder = _seeded(2, [])
    holder["sink"].append(_entry(2, "call_1"))
    assert _execute_on(holder)({"path": "a.txt"}, _ctx("call_2")) == client_module._ACK_ACCEPTED
    assert len(holder["sink"]) == 2

    # Positive control: the same entry, once reconciled, does suppress the repeat.
    control = _seeded(2, ["call_1"])
    control["sink"].append(_entry(2, "call_1"))
    assert _execute_on(control)({"path": "a.txt"}, _ctx("call_2")) == client_module._ACK_DUPLICATE
    assert len(control["sink"]) == 1


def test_each_run_of_a_nudged_turn_gets_its_own_generation():
    """Test 10: `_run_once` bumps the generation per run, so a nudged turn advances it twice."""
    client, sdk = _sdk_client()
    _sequenced(sdk, ["I'll read the file.", _CALL])
    client.chat.completions.create(
        model="grok-4.7",
        messages=[{"role": "user", "content": "apply the fix note"}],
        tools=[_tool()],
        _cursor_session_scope="scope-gen-nudged",
    )
    nudged = next(iter(client_module._SESSIONS.values())).tool_holder
    assert nudged["gen"] == 2
    assert isinstance(nudged["event_ids"], set)

    # Control: a turn with no nudge is one run, so a per-turn bump or a constant 2 would differ.
    client_module._reset_session_registry()
    _sequenced(sdk, ["All done."])
    client.chat.completions.create(
        model="grok-4.7",
        messages=[{"role": "user", "content": "apply the fix note"}],
        tools=[_tool()],
        _cursor_session_scope="scope-gen-plain",
    )
    assert next(iter(client_module._SESSIONS.values())).tool_holder["gen"] == 1


def test_holder_without_the_new_keys_survives_the_dedupe_predicate():
    """Test 11: a hand-built holder with neither new key must not KeyError in the predicate."""
    holder = _plain()
    execute = _execute_on(holder)
    assert execute({"path": "a.txt"}, _ctx("call_1")) == client_module._ACK_ACCEPTED
    # The sink is non-empty now, so this second identical call enters the predicate.
    assert execute({"path": "a.txt"}, _ctx("call_2")) == client_module._ACK_ACCEPTED
    assert len(holder["sink"]) == 2


def test_plan_with_zero_callbacks_is_an_error():
    client, sdk = _sdk_client()
    agent = _sequenced(sdk, ["I'll read the file.", "I'll read the file now."])
    with pytest.raises(client_module.CursorSDKRunError) as exc_info:
        _create(client)
    assert exc_info.value.error_code == "cursor_plan_only"
    assert exc_info.value.status_code == 400
    assert exc_info.value.body == {"error": {"code": "cursor_plan_only"}}
    assert len(agent.messages) == 2


def test_stream_emits_native_tool_call_deltas():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    try:
        chunks = list(
            client.chat.completions.create(
                model="grok-4.7",
                messages=[{"role": "user", "content": "CALL hermes_ping please"}],
                tools=[_tool("hermes_ping")],
                stream=True,
            )
        )
    finally:
        client.close()
    assert sdk.tool_callbacks == [client_module._ACK_ACCEPTED]
    assert _stream_text(chunks) == "hello"
    _assert_held_text_precedes_call_deltas(chunks)
    calls = _stream_calls(chunks)
    assert calls and calls[0].function.name == "hermes_ping"
    assert calls[0].function.arguments == '{"token":"probe-1"}'
    finals = [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason]
    assert finals == ["tool_calls"]


def test_toolset_change_recreate_resends_transcript():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    try:
        client.chat.completions.create(
            model="grok-4.7",
            messages=[{"role": "user", "content": "first-user"}],
            tools=[_tool("read_file")],
            _cursor_session_scope="scope-toolset",
        )
        client.chat.completions.create(
            model="grok-4.7",
            messages=[
                {"role": "user", "content": "first-user"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "second-user"},
            ],
            tools=[_tool("write_file")],
            _cursor_session_scope="scope-toolset",
        )
    finally:
        client.close()
    assert len(sdk.agents) == 2
    assert sdk.agents[0].closed
    resent = sdk.agents[1].messages[0]["text"]
    assert '"content":"first-user"' in resent
    assert '"content":"second-user"' in resent


# --- streaming: hold the reply until the turn decides, then emit in a fixed order ---


def _stream_agent(sdk, scripts):
    """Agent whose successive sends stream the given SDK events, one script per turn."""
    agent = FakeAgent(sdk)
    runs = list(scripts)

    def send(message):
        agent.messages.append(message)
        return ScriptedRun(sdk.result, runs.pop(0) if runs else [])

    agent.send = send

    def create(options):
        sdk.options.append(options)
        return agent

    sdk.create_agent = create
    return agent


def _streamed(client, tools=("read_file",), messages=None, **kwargs):
    return client.chat.completions.create(
        model="grok-4.7",
        messages=messages or [{"role": "user", "content": "go"}],
        tools=[_tool(name) for name in tools],
        stream=True,
        **kwargs,
    )


def test_plan_only_stream_emits_no_content_before_the_raise():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = ""
    _stream_agent(
        sdk,
        [[_assistant_event("I'll read the file.")], [_assistant_event("I'll read it now.")]],
    )
    seen = []
    with pytest.raises(client_module.CursorSDKRunError):
        for chunk in _streamed(client):
            seen.append(chunk)
    client.close()
    assert seen == []


def test_stream_flushes_after_final_marker():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = ""
    _stream_agent(
        sdk,
        [[_assistant_event("I'll read the file.")], [_assistant_event("FINAL: nothing to change.")]],
    )
    try:
        chunks = list(_streamed(client))
    finally:
        client.close()
    assert _stream_text(chunks) == "nothing to change."
    finals = [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason]
    assert finals == ["stop"]


def test_short_non_plan_streamed_reply_flushes_and_stops():
    """No nudge was sent, so the hold releases at run end and the turn completes."""
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = ""
    _stream_agent(sdk, [[_assistant_event("Done.")]])
    try:
        chunks = list(_streamed(client))
    finally:
        client.close()
    assert _stream_text(chunks) == "Done."
    finals = [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason]
    assert finals == ["stop"]


def test_nudged_long_reply_without_marker_raises_with_no_content():
    """After a nudge the hold never releases on length: the raise precedes every delta."""
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = ""
    _stream_agent(
        sdk,
        [[_assistant_event("I'll read the file.")], [_assistant_event("x" * 900)]],
    )
    seen = []
    with pytest.raises(client_module.CursorSDKRunError):
        for chunk in _streamed(client):
            seen.append(chunk)
    client.close()
    assert seen == []


def test_long_reply_releases_hold_before_run_end():
    """A pre-nudge reply over the 800-char intent cap is not an announcement; it streams."""
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = ""
    long_text = "A" * 900
    gate = threading.Event()

    def finish(run):
        assert gate.wait(10)
        return _assistant_event("")

    _stream_agent(sdk, [[_assistant_event(long_text), finish]])
    iterator = iter(_streamed(client))
    holder = {}

    def pull_first():
        holder["chunk"] = next(iterator, None)

    worker = threading.Thread(target=pull_first, daemon=True)
    worker.start()
    try:
        worker.join(2)
        assert not worker.is_alive(), "a >800-char reply was held past the release valve"
        assert holder["chunk"].choices[0].delta.content == long_text
        gate.set()
        rest = list(iterator)
    finally:
        gate.set()
        client.close()
        worker.join(5)
    assert _stream_text([holder["chunk"]]) + _stream_text(rest) == long_text


def test_nudged_reply_native_call_flushes_and_finishes():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = ""

    def call_event(run):
        entry = sdk.options[-1]["local"]["custom_tools"]["hermes_ping"]
        sdk.tool_callbacks.append(
            entry["execute"]({"token": "probe-1"}, SimpleNamespace(tool_call_id="call_probe"))
        )
        return SimpleNamespace(type="tool_call", call_id="call_probe\nfc_0")

    _stream_agent(
        sdk,
        [[_assistant_event("I'll check.")], [_assistant_event("Checking now."), call_event]],
    )
    try:
        chunks = list(_streamed(client, tools=("hermes_ping",)))
    finally:
        client.close()
    assert _stream_text(chunks) == "Checking now."  # the announcement is dropped at the boundary
    _assert_held_text_precedes_call_deltas(chunks)
    calls = _stream_calls(chunks)
    assert [call.function.name for call in calls] == ["hermes_ping"]
    finals = [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason]
    assert finals == ["tool_calls"]


def test_nudged_reply_marker_is_stripped_before_native_call_deltas():
    """A nudged reply that prefixes FINAL: AND calls must not leak the token to chunks."""
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = ""

    def call_event(run):
        entry = sdk.options[-1]["local"]["custom_tools"]["hermes_ping"]
        sdk.tool_callbacks.append(
            entry["execute"]({"token": "probe-1"}, SimpleNamespace(tool_call_id="call_probe"))
        )
        return SimpleNamespace(type="tool_call", call_id="call_probe\nfc_0")

    _stream_agent(
        sdk,
        [[_assistant_event("I'll check.")], [_assistant_event("FINAL: Checking now."), call_event]],
    )
    try:
        chunks = list(_streamed(client, tools=("hermes_ping",)))
    finally:
        client.close()
    content = _stream_text(chunks)
    assert content == "Checking now."
    assert "FINAL:" not in content


def test_run_timeout_is_configurable_and_defaults_to_thirty_minutes(monkeypatch):
    """The cap has a stated default and one lever; a malformed value cannot disable it."""
    monkeypatch.delenv("HERMES_CURSOR_RUN_TIMEOUT_SECONDS", raising=False)
    assert client_module._run_timeout_seconds() == 1800.0
    monkeypatch.setenv("HERMES_CURSOR_RUN_TIMEOUT_SECONDS", "45")
    assert client_module._run_timeout_seconds() == 45.0
    monkeypatch.setenv("HERMES_CURSOR_RUN_TIMEOUT_SECONDS", "not-a-number")
    assert client_module._run_timeout_seconds() == 1800.0
    monkeypatch.setenv("HERMES_CURSOR_RUN_TIMEOUT_SECONDS", "0")
    assert client_module._run_timeout_seconds() == 1800.0


class StalledRun(FakeRun):
    """A run that never reaches the SDK: stream() and wait() block until released."""

    def __init__(self, result, started, release):
        super().__init__(result)
        self.started = started
        self.release = release

    def stream(self):
        self.started.set()
        self.release.wait(5)
        yield _assistant_event("")

    def wait(self):
        self.release.wait(5)
        return self._result


def test_stalled_sdk_run_raises_the_timeout_instead_of_hanging(monkeypatch):
    """A call that never returns must fail the turn, not hold it open forever."""
    monkeypatch.setenv("HERMES_CURSOR_RUN_TIMEOUT_SECONDS", "0.2")
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    started, release = threading.Event(), threading.Event()
    run = StalledRun(sdk.result, started, release)
    agent = FakeAgent(sdk)
    agent.send = lambda message: run
    sdk.create_agent = lambda options: agent
    began = time.monotonic()
    try:
        with pytest.raises(client_module.CursorSDKRunError) as exc_info:
            client.chat.completions.create(
                model="grok-4.7",
                messages=[{"role": "user", "content": "hi"}],
                _cursor_session_scope="scope-stalled",
            )
        elapsed = time.monotonic() - began
    finally:
        release.set()
        client.close()
    assert exc_info.value.error_code == "cursor_run_timeout"
    assert exc_info.value.status_code == 408
    assert "HERMES_CURSOR_RUN_TIMEOUT_SECONDS" in str(exc_info.value)
    assert elapsed < 3, "the stall outlived its cap"
    assert started.is_set() and run.cancelled
    session = next(iter(client_module._SESSIONS.values()))
    assert session.agent is None and session.history == []
    assert not session.lock.locked()


def test_stalled_streamed_run_raises_before_any_content(monkeypatch):
    """The stall raise lands inside the held reply, so no content delta softens it."""
    monkeypatch.setenv("HERMES_CURSOR_RUN_TIMEOUT_SECONDS", "0.2")
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = ""
    release = threading.Event()

    def stall(run):
        release.wait(5)
        return _assistant_event("")

    _stream_agent(sdk, [[_assistant_event("I'll read the file."), stall]])
    seen = []
    try:
        with pytest.raises(client_module.CursorSDKRunError) as exc_info:
            for chunk in _streamed(client):
                seen.append(chunk)
    finally:
        release.set()
        client.close()
    assert exc_info.value.error_code == "cursor_run_timeout"
    assert seen == []


def test_stream_heartbeats_while_holding(monkeypatch):
    """A held stream still emits chunk-shaped keepalives (content=None) well before the kill."""
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = ""
    clock = {"t": 0.0}
    monkeypatch.setattr(
        client_module, "time", SimpleNamespace(monotonic=lambda: clock["t"])
    )

    def first(run):
        clock["t"] = 61.0
        return _assistant_event("I'll read the file.")

    def second(run):
        clock["t"] = 62.0
        return _assistant_event("")

    _stream_agent(sdk, [[first, second], [_assistant_event("I'll read it now.")]])
    seen = []
    with pytest.raises(client_module.CursorSDKRunError):
        for chunk in _streamed(client):
            seen.append(chunk)
    client.close()
    beats = [c for c in seen if c.choices and c.choices[0].delta.content is None]
    assert len(beats) == 1
    assert not [c for c in seen if c.choices and c.choices[0].delta.content]
