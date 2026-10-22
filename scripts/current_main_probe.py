"""Offline current-main provider seam probe."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_managed_stream(profile) -> None:
    """Exercise the real Relay consumer, not only provider registration."""
    from types import SimpleNamespace
    import threading
    from agent import relay_llm, relay_runtime
    from agent.chat_completion_helpers_relay import RelayChatAccumulator

    client = profile.create_client(api_key="offline-probe-token")
    calls = []
    allow_finish = threading.Event()
    completed = threading.Event()

    def send(message):
        import asyncio
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise AssertionError("SDK work must run off the Relay event loop")
        calls.append(message)
        return Run()

    class Run:
        def stream(self):
            yield SimpleNamespace(type="assistant", message=SimpleNamespace(content=[SimpleNamespace(text="relay-")]))
            assert allow_finish.wait(5), "Relay withheld text until completion"
            yield SimpleNamespace(type="assistant", message=SimpleNamespace(content=[SimpleNamespace(text="ok")]))

        def wait(self):
            completed.set()
            return SimpleNamespace(result="relay-ok", status="FINISHED", usage=None)

        def cancel(self):
            allow_finish.set()

    client._create_agent = lambda **kw: SimpleNamespace(send=send, close=lambda: None)
    coordinator = relay_runtime.SESSION_COORDINATOR
    lease = coordinator.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(), session_id="relay-probe", platform="cli")
    turn = coordinator.begin_turn(lease, turn_id="turn-1", task_id="probe")
    lease.host.retain_managed_execution("cursor-probe")
    accumulator = RelayChatAccumulator()
    stream = None
    try:
        stream = relay_llm.stream(
            {"model": "auto", "messages": [{"role": "user", "content": "hello"}], "stream": True},
            lambda request: client.chat.completions.create(**request),
            session_id="relay-probe", name="cursor", model_name="auto",
            finalizer=accumulator.finalize, on_chunk=accumulator.observe,
            completed_response_predicate=lambda value: hasattr(value, "choices"),
            metadata={"api_mode": "chat_completions"},
        )
        first = next(iter(stream))
        assert first.choices[0].delta.content == "relay-"
        assert not completed.is_set()
        allow_finish.set()
        chunks = [first, *stream]
        assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == "relay-ok"
        assert chunks[-1].usage.total_tokens == 0
        assert len(calls) == 1
    finally:
        allow_finish.set()
        if stream is not None:
            stream.close()
        client.close()
        lease.host.release_managed_execution("cursor-probe")
        coordinator.end_turn(turn, outcome="success")
        coordinator.release_conversation(lease)
        relay_runtime._reset_for_tests()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="hermes-cursor-probe-") as raw_home:
        home = Path(raw_home)
        target = home / "plugins" / "cursor-provider"
        target.parent.mkdir(parents=True)
        shutil.copytree(ROOT / "provider", target)
        os.environ["HERMES_HOME"] = str(home)
        os.environ["CURSOR_API_KEY"] = "offline-probe-token"

        import providers

        profile = providers.get_provider_profile("cursor")
        assert profile is not None
        assert profile.auth_type == "api_key"
        assert profile.auth_handler is None
        assert profile.refresh_credential is None
        assert profile.env_vars == ("CURSOR_API_KEY",)
        assert profile.build_api_kwargs_extras(
            session_id="session-1", cache_scope_id="scope-1"
        ) == ({}, {"_cursor_session_scope": "scope-1"})
        assert profile.build_api_kwargs_extras(session_id="session-1") == (
            {}, {"_cursor_session_scope": "session-1"}
        )
        assert profile.build_api_kwargs_extras() == ({}, {})

        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(
            requested="cursor",
            explicit_api_key="offline-probe-token",
            explicit_base_url="https://api.cursor.com",
            target_model="auto",
        )
        assert runtime["provider"] == "cursor"
        assert runtime["api_key"] == "offline-probe-token"
        assert runtime["base_url"] == "https://api.cursor.com"
        assert runtime["api_mode"] == "chat_completions"

        from run_agent import AIAgent
        from agent.chat_completion_helpers import build_api_kwargs

        agent = AIAgent(
            api_key=runtime["api_key"],
            base_url=runtime["base_url"],
            provider="cursor",
            model="auto",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        setattr(agent, "session_id", "offline-session")
        request = build_api_kwargs(
            agent, [{"role": "user", "content": "hello"}], []
        )
        assert request.get("_cursor_session_scope")
        agent.close()

        client = profile.create_client(api_key=runtime["api_key"], base_url=runtime["base_url"])
        assert client is not None
        assert client.chat.completions.create
        assert client.api_key == "offline-probe-token"
        assert client.HERMES_SKIP_TRANSPORT_WRAP
        assert client.HERMES_SKIP_ASYNC_WRAP
        error_module = sys.modules[type(client).__module__]
        timeout_error = error_module.CursorSDKRunError(
            "offline deadline", status_code=408, error_code="cursor_run_timeout"
        )
        from agent.error_classifier import FailoverReason, classify_api_error

        timeout_verdict = classify_api_error(timeout_error)
        assert timeout_verdict.reason is FailoverReason.timeout
        assert timeout_verdict.retryable and not timeout_verdict.should_fallback
        client.close()

        from agent.auxiliary_client import resolve_provider_client

        aux_client, aux_model = resolve_provider_client(
            "cursor",
            model="auto",
            async_mode=True,
            explicit_api_key="offline-probe-token",
            explicit_base_url="https://api.cursor.com",
        )
        assert aux_client is not None
        assert aux_model == "auto"
        assert aux_client.HERMES_SKIP_ASYNC_WRAP
        aux_client.close()
        verify_managed_stream(profile)

        print(json.dumps({
            "ok": True,
            "managed_relay_stream": True,
            "hermes_source": str(Path(providers.__file__).resolve()),
            "provider": runtime["provider"],
            "api_mode": runtime["api_mode"],
            "base_url": runtime["base_url"],
        }, sort_keys=True))


if __name__ == "__main__":
    main()
