"""Deadline regressions over the real cursor-sdk Run/Agent stream machinery.

The rest of the suite drives hand-written FakeRun objects, which cannot show whether a
blocking SDK read is actually bounded or whether ``Run.wait()`` uses the SDK's cached
terminal result. These tests keep the real SDK classes but replace only the transport,
so the lazy Send stream, Run priming, cached wait, and WaitLiveRun fallback all run.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("cursor_sdk")

from cursor_sdk import Client as SDKClient  # noqa: E402
from cursor_sdk import Run as SDKRun  # noqa: E402

from hermes_cursor_provider import client as client_module  # noqa: E402
from hermes_cursor_provider.client import CursorSDKClient, CursorSDKRunError  # noqa: E402


def _assistant(text):
    return {"sdkMessage": {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}}


def _result(text):
    return {
        "result": {
            "result": {
                "runId": "run-1",
                "agentId": "agent-1",
                "status": "FINISHED",
                "result": text,
            }
        }
    }


class _Transport:
    """Real SDK transport seam with scripted Send streams and recorded unary calls."""

    def __init__(self, script):
        self.script = script
        self.calls = []
        self.release = threading.Event()
        self.cancelled = threading.Event()
        self.send_blocked = threading.Event()
        self.second_read = threading.Event()
        self.wait_started = threading.Event()
        self.closed = threading.Event()

    def with_options(self, **_):
        return self

    def unary(self, service, method, message):
        self.calls.append((method, dict(message)))
        if method == "CreateAgent":
            return {"agentId": "agent-1", "model": {"id": "auto"}}
        if method == "CancelRun":
            self.cancelled.set()
            self.release.set()
            return {}
        if method == "WaitLiveRun":
            self.wait_started.set()
            self.release.wait(10)
            return {"result": {"runId": "run-1", "agentId": "agent-1", "status": "FINISHED", "result": "late"}}
        return {}

    def stream(self, service, method, message):
        assert method == "Send", method
        return self.script(self)

    def close(self):
        self.closed.set()

    def methods(self):
        return [method for method, _ in self.calls]


def _install_real_sdk(monkeypatch, tmp_path, transport):
    """Install a launcher that builds the real SDK from the plugin's bridge options."""
    holder = SimpleNamespace(sdk=None)

    def launch_bridge(**kwargs):
        holder.sdk = SDKClient(
            transport=transport,
            allow_api_key_env_fallback=kwargs["allow_api_key_env_fallback"],
        )
        return holder.sdk

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr(
        client_module,
        "_load_cursor_client",
        lambda: SimpleNamespace(launch_bridge=launch_bridge),
    )
    return holder


@pytest.fixture
def cleanup_registry():
    yield
    client_module._reset_session_registry()


def test_real_sdk_stream_read_hits_deadline_and_cancels_run(monkeypatch, tmp_path, cleanup_registry):
    def script(transport):
        yield _assistant("hel")
        transport.second_read.set()
        transport.release.wait(10)
        yield _assistant("lo")
        yield _result("helo")

    transport = _Transport(script)
    _install_real_sdk(monkeypatch, tmp_path, transport)
    # A CURSOR_API_KEY inherited from the process must never become this run's account.
    monkeypatch.setenv("CURSOR_API_KEY", "env-wrong-account")
    monkeypatch.setattr(client_module, "_run_timeout_seconds", lambda: 0.05)
    client = CursorSDKClient(api_key="crsr_test")
    started = time.monotonic()
    with pytest.raises(CursorSDKRunError) as excinfo:
        client.chat.completions.create(
            model="auto", messages=[{"role": "user", "content": "hi"}]
        )
    elapsed = time.monotonic() - started

    assert excinfo.value.status_code == 408
    assert excinfo.value.error_code == "cursor_run_timeout"
    assert elapsed < 1.0, elapsed
    assert transport.second_read.wait(1), "the deadline did not interrupt the second read"
    assert transport.cancelled.wait(1), "the real Run was not cancelled"
    assert transport.methods().count("CancelRun") == 1, "the run was cancelled more than once"
    assert "WaitLiveRun" not in transport.methods()
    assert all("env-wrong-account" not in repr(call) for call in transport.calls), (
        "a run-scoped call fell back to the process CURSOR_API_KEY"
    )
    client.close()


def test_real_sdk_completed_stream_uses_cached_wait(monkeypatch, tmp_path, cleanup_registry):
    def script(transport):
        yield _assistant("hello")
        yield _result("hello")

    transport = _Transport(script)
    _install_real_sdk(monkeypatch, tmp_path, transport)
    monkeypatch.setattr(client_module, "_run_timeout_seconds", lambda: 1.0)
    client = CursorSDKClient(api_key="crsr_test")
    response = client.chat.completions.create(
        model="auto", messages=[{"role": "user", "content": "hi"}]
    )

    assert response.choices[0].message.content == "hello"
    assert "WaitLiveRun" not in transport.methods(), "a drained stream should not hit the unary wait"
    assert "CancelRun" not in transport.methods(), "a completed run was cancelled"
    client.close()


def test_real_sdk_wait_live_run_path_is_bounded(monkeypatch, tmp_path, cleanup_registry):
    transport = _Transport(lambda _: iter(()))
    _install_real_sdk(monkeypatch, tmp_path, transport)
    monkeypatch.setattr(client_module, "_run_timeout_seconds", lambda: 0.05)
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    run = SDKRun(client=sdk, run_id="run-1", agent_id="agent-1")
    agent = SimpleNamespace(send=lambda message: run, close=lambda: None)
    sdk.create_agent = lambda options: agent
    started = time.monotonic()
    with pytest.raises(CursorSDKRunError) as excinfo:
        client.chat.completions.create(
            model="auto", messages=[{"role": "user", "content": "hi"}]
        )
    elapsed = time.monotonic() - started

    assert excinfo.value.status_code == 408
    assert elapsed < 1.0, elapsed
    assert transport.wait_started.is_set(), "the real Run.wait() unary path was not exercised"
    assert transport.cancelled.wait(1)
    assert "CancelRun" in transport.methods()
    client.close()


def test_real_sdk_stream_stays_lazy_between_pulls(monkeypatch, tmp_path, cleanup_registry):
    def script(transport):
        yield _assistant("hel")
        transport.second_read.set()
        transport.release.wait(10)
        yield _assistant("lo")
        yield _result("helo")

    transport = _Transport(script)
    _install_real_sdk(monkeypatch, tmp_path, transport)
    monkeypatch.setattr(client_module, "_run_timeout_seconds", lambda: 1.0)
    client = CursorSDKClient(api_key="crsr_test")
    stream = client.chat.completions.create(
        model="auto", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    iterator = iter(stream)
    first = next(iterator)

    assert first.choices[0].delta.content == "hel"
    assert not transport.second_read.is_set(), "a fragment was read before the consumer asked for it"
    transport.release.set()
    rest = list(iterator)
    assert _text([first, *rest]) == "hello"
    client.close()


def _text(chunks):
    return "".join(
        chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices
    )


def test_send_read_is_bounded_and_late_run_is_cancelled(monkeypatch, tmp_path, cleanup_registry):
    def script(transport):
        transport.send_blocked.set()
        transport.release.wait(10)
        yield _assistant("late")
        yield _result("late")

    transport = _Transport(script)
    _install_real_sdk(monkeypatch, tmp_path, transport)
    monkeypatch.setattr(client_module, "_run_timeout_seconds", lambda: 0.05)
    client = CursorSDKClient(api_key="crsr_test")
    started = time.monotonic()
    with pytest.raises(CursorSDKRunError) as excinfo:
        client.chat.completions.create(
            model="auto", messages=[{"role": "user", "content": "hi"}]
        )
    elapsed = time.monotonic() - started

    assert excinfo.value.status_code == 408
    assert elapsed < 1.0, elapsed
    assert transport.send_blocked.wait(1)
    transport.release.set()
    deadline = time.monotonic() + 2
    while not transport.cancelled.is_set() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert transport.cancelled.is_set(), "a run created after the send deadline was not cancelled"
    assert transport.methods().count("CancelRun") == 1, "the late run was cancelled more than once"
    client.close()


def test_read_outcome_hands_off_a_deadline_result_exactly_once():
    """The caller/worker handshake must never lose a result at the deadline."""
    returned = client_module._ReadOutcome()
    assert returned.publish(value="run") is True
    assert returned.take() is True
    assert returned.value == "run"

    # The wait can expire just before the worker publishes; the result must still be
    # collected rather than dropped between the two checks.
    raced = client_module._ReadOutcome()
    raced.wait(0)
    assert raced.publish(value="run") is True
    assert raced.take() is True
    assert raced.value == "run"

    abandoned = client_module._ReadOutcome()
    abandoned.wait(0)
    assert abandoned.take() is False, "the caller did not claim the deadline"
    assert abandoned.publish(value="run") is False, (
        "an abandoned read published its result instead of handing it to late cleanup"
    )


def test_cleanup_that_blocks_does_not_extend_the_deadline(monkeypatch, cleanup_registry):
    release = threading.Event()

    class HangingWaitRun:
        status = "running"

        def __init__(self):
            self.cancelled = False

        def stream(self):
            yield _assistant("hi")

        def wait(self):
            release.wait(10)
            return SimpleNamespace(result="never", status="FINISHED")

        def cancel(self):
            release.wait(10)
            self.cancelled = True

    run = HangingWaitRun()

    class HangingCloseAgent:
        def send(self, message):
            return run

        def close(self):
            release.wait(10)

    agent = HangingCloseAgent()
    monkeypatch.setattr(client_module, "_run_timeout_seconds", lambda: 0.05)
    client = CursorSDKClient(api_key="crsr_test")
    client._create_agent = lambda **_: agent
    monkeypatch.setattr(client, "_sdk_client", lambda: SimpleNamespace())
    started = time.monotonic()
    try:
        with pytest.raises(CursorSDKRunError) as excinfo:
            client.chat.completions.create(
                model="auto", messages=[{"role": "user", "content": "hi"}]
            )
        elapsed = time.monotonic() - started
    finally:
        release.set()
        client.close()
    assert excinfo.value.status_code == 408
    assert elapsed < 1.0, f"cleanup extended the 0.05s deadline to {elapsed:.3f}s"
    deadline = time.monotonic() + 2
    while not run.cancelled and time.monotonic() < deadline:
        time.sleep(0.01)
    assert run.cancelled, "the abandoned cancel worker never reached the run"
