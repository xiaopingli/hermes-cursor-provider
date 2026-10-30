"""OpenAI-compatible facade over Cursor's official Python SDK."""
from __future__ import annotations

import atexit
import asyncio
import base64
import hashlib
import importlib
import json
import logging
import math
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, cast

CURSOR_API_URL = "https://api.cursor.com"
_MAX_IMAGES = 4
_MAX_IMAGE_BASE64_CHARS = 28_000_000
# Wall-clock cap for one Cursor SDK run. The cap covers the run's Send stream and its
# result wait; the SDK offers no total-deadline option, so the plugin enforces it.
_DEFAULT_RUN_TIMEOUT_SECONDS = 1800.0
_MIN_RUN_TIMEOUT_SECONDS = 0.05
_MAX_RUN_TIMEOUT_SECONDS = 86_400.0
# A completed run's ``wait()`` returns from the SDK's cache; this small grace keeps a
# cached result collectable when the stream consumed the cap exactly.
_WAIT_COMPLETION_GRACE_SECONDS = 0.05
# Best-effort cancel/close calls get a short, cap-scaled budget so cleanup cannot add
# another full transport timeout to a turn that has already hit its deadline.
_CLEANUP_MIN_SECONDS = 0.1
_CLEANUP_MAX_SECONDS = 2.0

_LOG = logging.getLogger("hermes_cursor_provider")
_FALSEY = {"0", "false", "no", "off", ""}
_SECRET_KEY_RE = re.compile(r"(pass|secret|token|key|auth|cookie|credential|^text$|^value$|^values$|^fields$|^input$)", re.I)
# Raw payload fields are never logged, even if a caller passes them by mistake.
_DEBUG_RAW_FIELDS = {"args", "arguments", "result", "block"}
# Cursor-native MCP wrapper tool names; their args carry the real tool name.
_NATIVE_WRAPPERS = {"callmcptool", "call_mcp_tool", "mcp", "use_mcp_tool"}
_NATIVE_INTERCEPT_FORCED_LOGGED = False
_INERT_ACK = (
    "Recorded for Hermes. Hermes decides whether to run it after this turn; "
    "the result, if any, arrives in the next message. Do not repeat this call."
)


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() not in _FALSEY


def _mask(value: Any, depth: int = 0) -> Any:
    """Bounded, secret-masked copy of a value for the debug event log."""
    if depth > 4:
        return "…"
    if isinstance(value, dict):
        return {
            str(k): ("***" if _SECRET_KEY_RE.search(str(k)) else _mask(v, depth + 1))
            for k, v in list(value.items())[:40]
        }
    if isinstance(value, (list, tuple)):
        return [_mask(v, depth + 1) for v in list(value)[:20]]
    if isinstance(value, str):
        return value[:300] + ("…" if len(value) > 300 else "")
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return str(value)[:300]


def _debug_event(kind: str, **fields: Any) -> None:
    """Append one JSON line to <HERMES_HOME>/logs/cursor-provider-events.jsonl.

    Off unless HERMES_CURSOR_DEBUG_EVENTS=1. Callers pass bounded metadata only
    (names, statuses, ids, argument keys); raw arguments, results, and blocks are
    never logged. Values are truncated and secret-masked as a second line of defense."""
    if not _flag("HERMES_CURSOR_DEBUG_EVENTS", False):
        return
    try:
        path = Path(_home_key()) / "logs" / "cursor-provider-events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        record = {
            "ts": round(time.time(), 3),
            "kind": kind,
            **{k: _mask(v) for k, v in fields.items() if k not in _DEBUG_RAW_FIELDS},
        }
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            with suppress(OSError):
                os.chmod(path, 0o600)  # tighten logs created by older versions
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:  # logging must never break a turn
        pass


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).replace("-", "_").lower()


def match_allowed_name(name: Any, allowed: set[str]) -> str | None:
    """Map a model-written tool name onto an offered Hermes tool name, or None.

    Accepts exact names, namespace/prefix forms (functions.x, mcp__srv__x, mcp_x),
    CamelCase, and a unique ``*_name`` suffix match for multi-word names."""
    raw = str(name or "").strip()
    if not raw:
        return None
    if raw in allowed:
        return raw
    cands = [raw]
    for sep in (".", "/", ":", "__"):
        if sep in raw:
            cands.append(raw.split(sep)[-1])
    cands += [_snake(c) for c in list(cands)]
    cands = list(dict.fromkeys(cands))
    for cand in cands:
        if cand in allowed:
            return cand
        if cand.startswith("mcp_") and cand[4:] in allowed:
            return cand[4:]
    for cand in cands:
        if "_" in cand:
            hits = [a for a in allowed if a.endswith("_" + cand)]
            if len(hits) == 1:
                return hits[0]
    return None


class CursorSDKRunError(RuntimeError):
    """A terminal Cursor SDK run failure with Hermes-readable status."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        lowered = message.lower()
        default_status = 429 if "out of usage" in lowered or "increase limits" in lowered else 502
        self.status_code = status_code if status_code is not None else default_status
        self.error_code = error_code or "cursor_sdk_run_failed"


def _run_timeout_seconds() -> float:
    """How long one SDK run may take before the plugin stops waiting, in seconds.

    Default 1800 s (30 min): long enough for a legitimate large-context call, short
    enough to fail before an operator or a supervisor kills the whole turn. A malformed,
    non-finite, or non-positive value falls back to the default — the cap exists to stop
    the hang, so it cannot be switched off or turned into an instant timeout.
    """
    raw = os.environ.get("HERMES_CURSOR_RUN_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return _DEFAULT_RUN_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        _LOG.warning("cursor: ignoring malformed HERMES_CURSOR_RUN_TIMEOUT_SECONDS=%r", raw)
        return _DEFAULT_RUN_TIMEOUT_SECONDS
    if not math.isfinite(value) or value <= 0:
        _LOG.warning("cursor: ignoring invalid HERMES_CURSOR_RUN_TIMEOUT_SECONDS=%r", raw)
        return _DEFAULT_RUN_TIMEOUT_SECONDS
    return min(max(value, _MIN_RUN_TIMEOUT_SECONDS), _MAX_RUN_TIMEOUT_SECONDS)


def _run_timeout_error(cap: float, elapsed: float, phase: str) -> CursorSDKRunError:
    """The loud end of a stalled run: a retryable timeout Hermes can act on."""
    return CursorSDKRunError(
        f"Cursor SDK run exceeded its {cap:g}s deadline after {elapsed:g}s while {phase}; "
        "Hermes stopped waiting instead of hanging the turn. Raise "
        "HERMES_CURSOR_RUN_TIMEOUT_SECONDS if this workload legitimately needs longer.",
        status_code=408,
        error_code="cursor_run_timeout",
    )


class _ReadOutcome:
    """Exactly-once handoff between one bounded read and its caller.

    ``publish`` is the worker's claim and ``take`` is the caller's; both decide under
    one lock. A published value is returned to the caller. An abandoned read hands the
    value back to ``late`` cleanup. A result that lands at the deadline therefore
    cannot fall between the two: either the caller takes it, or the worker cleans it up.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._abandoned = False
        self.value: Any = None
        self.error: BaseException | None = None

    def publish(self, *, value: Any = None, error: BaseException | None = None) -> bool:
        """Publish the read's outcome; False means the caller already gave up."""
        with self._lock:
            if self._abandoned:
                return False
            self.value = value
            self.error = error
            self._done.set()
            return True

    def wait(self, timeout: float) -> None:
        self._done.wait(timeout)

    def take(self) -> bool:
        """Claim the published outcome, or abandon the read for ``late`` cleanup."""
        with self._lock:
            if self._done.is_set():
                return True
            self._abandoned = True
            return False


def _bounded_call(
    call: Any,
    *,
    deadline: float,
    cap: float,
    started: float,
    phase: str,
    grace: float = 0.0,
    late: Any = None,
) -> Any:
    """Run one blocking SDK operation on a daemon thread, bounded by ``deadline``.

    ``run.stream()`` is lazy: every ``next()`` and ``run.wait()`` is a separate blocking
    transport read. Bounding each read against the same deadline keeps delivery lazy and
    backpressure intact (exactly one read is in flight, and only when the consumer asks)
    while making the cap cover the real stream/wait path rather than only ``wait()``.

    At the deadline the caller raises immediately; the worker is abandoned (daemon
    thread) because a wedged transport read cannot be interrupted. ``late`` receives an
    operation result that arrives after the caller gave up, so a timed-out ``send()`` can
    still cancel the run it eventually created. ``_ReadOutcome`` guarantees that cleanup
    runs exactly once, even when the result lands while the caller is timing out.
    """
    outcome = _ReadOutcome()

    def _invoke() -> None:
        try:
            value = call()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            outcome.publish(error=exc)
        else:
            if not outcome.publish(value=value) and late is not None:
                try:
                    late(value)
                except Exception:  # cleanup must never mask the timeout
                    pass

    remaining = deadline - time.monotonic()
    if remaining <= 0 and grace <= 0:
        raise _run_timeout_error(cap, time.monotonic() - started, phase)
    # ponytail: one short-lived thread per blocking read. A persistent demand-driven
    # worker would cut thread churn, but add lifecycle state for no measured gain here.
    worker = threading.Thread(target=_invoke, name="cursor-sdk-read", daemon=True)
    worker.start()
    outcome.wait(max(remaining, grace))
    if not outcome.take():
        raise _run_timeout_error(cap, time.monotonic() - started, phase)
    if outcome.error is not None:
        raise outcome.error
    return outcome.value


def _cleanup_grace_seconds() -> float:
    """A short, cap-scaled budget for one best-effort cancel/close call."""
    return max(
        _CLEANUP_MIN_SECONDS,
        min(_CLEANUP_MAX_SECONDS, _run_timeout_seconds() / 1000.0),
    )


def _run_cleanup(call: Any, what: str) -> None:
    """Run one cleanup call without letting a wedged transport add a full timeout.

    Cleanup still happens on a worker so a hung ``CancelRun``/``CloseAgent`` cannot
    push a turn past its deadline. If it outlives the grace it is abandoned; the SDK's
    own unary timeout eventually reclaims the worker.
    """
    grace = _cleanup_grace_seconds()

    def _invoke() -> None:
        with suppress(Exception):
            call()

    worker = threading.Thread(target=_invoke, name="cursor-sdk-cleanup", daemon=True)
    worker.start()
    worker.join(grace)
    if worker.is_alive():
        _LOG.warning("cursor: %s did not finish within %gs; abandoning the call", what, grace)


def _cancel_run(run: Any) -> None:
    if run is not None:
        _run_cleanup(run.cancel, "cancel the Cursor run")


def _close_agent(agent: Any) -> None:
    if agent is not None:
        _run_cleanup(agent.close, "close the Cursor agent")


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(exclude_none=True))
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    if hasattr(value, "__dict__"):
        return _plain(vars(value))
    return value


def _secret(value: Any) -> str:
    getter = getattr(value, "get_secret_value", None)
    return str(getter() if callable(getter) else (value or "")).strip()


def _timeout_seconds(value: Any, default: float = 300.0) -> float:
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    for attr in ("read", "connect"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, (int, float)) and candidate > 0:
            return float(candidate)
    return default


def _content_text(content: Any) -> str:
    content = _plain(content)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                parts.append(str(part))
                continue
            if part.get("type") in {"image", "image_url", "input_image"} or part.get("image_url"):
                parts.append("[image attached separately]")
                continue
            text = part.get("text") or part.get("content")
            if text:
                parts.append(str(text))
        return "\n".join(parts)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    return str(content or "")


def _canonical_tool_calls(calls: Any) -> list[dict[str, Any]]:
    canonical: list[dict[str, Any]] = []
    for raw in calls if isinstance(calls, list) else []:
        call = _plain(raw)
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        arguments = function.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        canonical.append(
            {
                "id": str(call.get("id") or call.get("call_id") or ""),
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return canonical


def _conversation(messages: list[Any]) -> list[dict[str, Any]]:
    transcript: list[dict[str, Any]] = []
    for raw in messages:
        message = _plain(raw)
        if not isinstance(message, dict):
            continue
        item: dict[str, Any] = {
            "role": str(message.get("role") or "context"),
            "content": _content_text(message.get("content")),
        }
        for key in ("name", "tool_call_id"):
            if message.get(key):
                item[key] = str(message[key])
        calls = _canonical_tool_calls(message.get("tool_calls"))
        if calls:
            item["tool_calls"] = calls
        transcript.append(item)
    return transcript


def tool_specs(tools: list[Any] | None, tool_choice: Any = None) -> list[dict[str, Any]]:
    if tool_choice == "none":
        return []
    selected_name = ""
    choice = _plain(tool_choice)
    if isinstance(choice, dict):
        function = choice.get("function")
        if isinstance(function, dict):
            selected_name = str(function.get("name") or "").strip()
    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in tools or []:
        tool = _plain(raw)
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        if not name or name in seen or (selected_name and name != selected_name):
            continue
        parameters = function.get("parameters")
        specs.append(
            {
                "name": name,
                "description": str(function.get("description") or ""),
                "parameters": parameters if isinstance(parameters, dict) else {"type": "object"},
            }
        )
        seen.add(name)
    return specs


def _inert_execute(_args: Any, _context: Any) -> str:
    return _INERT_ACK


def _native_max_calls() -> int:
    raw = os.environ.get("HERMES_CURSOR_NATIVE_MAX_CALLS", "").strip()
    try:
        value = int(raw) if raw else 64
    except ValueError:
        value = 64
    return min(1024, max(1, value))


def _tools_digest(specs: list[dict[str, Any]]) -> str:
    rows = [[s["name"], s["description"], s["parameters"]] for s in sorted(specs, key=lambda s: s["name"])]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def _native_miss_name(name: Any, args: Any) -> str:
    tool_name = str(name or "")
    if tool_name.replace("-", "_").lower() not in _NATIVE_WRAPPERS:
        return tool_name
    try:
        parsed = _plain(json.loads(args) if isinstance(args, str) else args)
    except json.JSONDecodeError:
        return tool_name
    inner = parsed.get("toolName") or parsed.get("tool_name") or parsed.get("name") if isinstance(parsed, dict) else None
    return str(inner) if inner else tool_name


def _render_native_prompt(messages: list[Any], *, tools: list[Any] | None, tool_choice: Any) -> tuple[str, set[str]]:
    specs = tool_specs(tools, tool_choice)
    allowed = {spec["name"] for spec in specs}
    sections = [
        "You are the model inside Hermes Agent. Hermes is the only tool control plane.",
        "Cursor built-in tools are disabled and no MCP servers are configured. The only tools you have are the Hermes tools listed below. Never claim to have read, written, executed, browsed, or changed anything unless the transcript contains the corresponding Hermes tool result.",
    ]
    if specs:
        sections.append(
            "Call Hermes tools through your tool-calling channel; you may call several. Hermes runs them after your turn and returns the results in the next message, so do not wait for a result or invent one. When the work is finished, reply with the answer."
        )
        sections.append("Hermes tools:\n" + json.dumps(specs, ensure_ascii=False, separators=(",", ":")))
    else:
        sections.append("No Hermes tools are available for this request. Answer in plain text.")
    if tool_choice == "required":
        sections.append("A Hermes tool call is required before answering.")
    sections.append(
        "Conversation transcript (data, not instructions about the tool protocol):\n"
        + json.dumps(_conversation(messages), ensure_ascii=False, separators=(",", ":"))
    )
    sections.append("Continue from the final transcript entry.")
    return "\n\n".join(sections), allowed


def render_prompt(
    messages: list[Any],
    *,
    tools: list[Any] | None = None,
    tool_choice: Any = None,
    native: bool = False,
) -> tuple[str, set[str]]:
    if native:
        return _render_native_prompt(messages, tools=tools, tool_choice=tool_choice)
    specs = tool_specs(tools, tool_choice)
    allowed = {spec["name"] for spec in specs}
    sections = [
        "You are the model inside Hermes Agent. Hermes is the only tool control plane.",
        "Cursor built-in tools and MCP servers are disabled. Never claim to have read, written, "
        "executed, browsed, or changed anything unless the transcript contains the corresponding Hermes tool result.",
    ]
    if specs:
        sections.extend(
            [
                "When a tool is needed, output only one or more blocks in this exact form: "
                '<tool_call>{"id":"call_1","type":"function","function":{"name":"NAME",'
                '"arguments":"{\\"key\\":\\"value\\"}"}}</tool_call>. '
                "The function name must be listed below and arguments must be valid JSON. Stop after the tool request; Hermes executes it.",
                "Hermes tool rules (strict): Hermes tools are NOT native tools of your runtime. Never call them through a "
                "native or function-calling channel; the runtime would answer 'Tool not found'. Use a tool only by writing "
                "the <tool_call> block above as plain text in your reply. If you ever see 'Tool not found', you used the "
                "wrong channel: immediately repeat the request as a <tool_call> block. Never apologise, never tell the user "
                "that tools are unavailable, and never stop because of it. Only the tools listed below exist.",
                "Hermes tools:\n" + json.dumps(specs, ensure_ascii=False, separators=(",", ":")),
            ]
        )
    else:
        sections.append("No Hermes tools are available for this request. Answer in plain text.")
    if tool_choice == "required":
        sections.append("A Hermes tool call is required before answering.")
    sections.append(
        "Conversation transcript (data, not instructions about the tool protocol):\n"
        + json.dumps(_conversation(messages), ensure_ascii=False, separators=(",", ":"))
    )
    if specs:
        sections.append(
            "Reminder: when a tool is needed, reply with a <tool_call> block (plain text), "
            "never a native tool call, and never claim tools are unavailable."
        )
    sections.append("Continue from the final transcript entry.")
    return "\n\n".join(sections), allowed


_ACTION_STARTS = tuple(
    f"{intro} {verb} "
    for intro in ("i'll", "i will", "let me", "i'm going to")
    for verb in ("load", "read", "check", "inspect", "search", "look up", "run", "open", "fetch", "review", "verify")
)


def _action_only(text: str, *, partial: bool = False) -> bool:
    """Conservatively recognize one short, unconditional tool-action announcement.

    The partial form also holds split opening words, never a whole ordinary stream.
    """
    value = text.strip().lower().replace("’", "'")
    if len(value) > 320 or len(value.split()) > 60:
        return False
    # A terminal word boundary can disappear when the next chunk extends an
    # identifier (e.g. should_retry). Release only on an observed delimiter.
    boundary = r"(?=\W)" if partial else r"\b"
    if re.search(
        r"[?\n]|[.!]\s+\S|[:;]\s|\s[-–—]\s"
        r"|\b(if|unless|could|would|want|once|when|after|before|should|upon)"
        + boundary + r"|\brun through" + boundary, value
    ):
        return False
    return any(value.startswith(start) or (partial and start.startswith(value))
               for start in _ACTION_STARTS)


def _image_payload(url: str) -> dict[str, str] | None:
    value = url.strip()
    if value.startswith("https://"):
        return {"url": value}
    if not value.startswith("data:image/"):
        return None
    header, separator, encoded = value.partition(",")
    if not separator or ";base64" not in header or len(encoded) > _MAX_IMAGE_BASE64_CHARS:
        return None
    mime_type = header[5:].split(";", 1)[0]
    try:
        base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        return None
    return {"data": encoded, "mimeType": mime_type}


def extract_images(messages: list[Any]) -> list[dict[str, str]]:
    images: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in messages:
        message = _plain(raw)
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            image = part.get("image_url")
            if isinstance(image, dict):
                url = image.get("url")
            elif isinstance(image, str):
                url = image
            else:
                url = part.get("url") if part.get("type") in {"image", "image_url", "input_image"} else None
            if not isinstance(url, str) or url in seen:
                continue
            payload = _image_payload(url)
            if payload is None:
                continue
            images.append(payload)
            seen.add(url)
            if len(images) >= _MAX_IMAGES:
                return images
    return images


def _tool_call_from_payload(payload_text: str, allowed_names: set[str], ordinal: int) -> Any | None:
    """Parse one <tool_call> JSON body into a Hermes tool call, or None when unusable."""
    try:
        payload = json.loads(payload_text)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    function = payload.get("function")
    if not isinstance(function, dict):
        # Lenient: {"name": ..., "arguments"|"parameters"|"args"|"input": ...} without the wrapper.
        keys = [k for k in ("arguments", "parameters", "args", "input") if k in payload]
        if "name" not in payload or not keys:
            return None
        function = {"name": payload["name"], "arguments": payload[keys[0]]}
    name = match_allowed_name(function.get("name"), allowed_names) or ""
    if not name:
        return None
    arguments = function.get("arguments", "{}")
    if arguments is None:
        arguments = "{}"
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    try:
        decoded = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, dict):
        return None
    call_id = str(payload.get("id") or f"cursor_call_{ordinal}").strip()
    return SimpleNamespace(
        id=call_id,
        call_id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _tool_calls_from_block(body: str, allowed_names: set[str], first_ordinal: int) -> list[Any]:
    """Parse a <tool_call> body that holds one call, several concatenated JSON calls, or a JSON array.

    Models often emit parallel calls as consecutive JSON objects inside a single block; the whole
    body is not valid JSON then, so each object is decoded in turn and kept when it is a usable call."""
    single = _tool_call_from_payload(body, allowed_names, first_ordinal)
    if single is not None:
        return [single]
    decoder = json.JSONDecoder()
    calls: list[Any] = []
    index, length = 0, len(body)
    while index < length:
        while index < length and (body[index].isspace() or body[index] == ","):
            index += 1
        if index >= length:
            break
        try:
            value, index = decoder.raw_decode(body, index)
        except json.JSONDecodeError:
            break
        for item in value if isinstance(value, list) else [value]:
            if not isinstance(item, dict):
                continue
            call = _tool_call_from_payload(
                json.dumps(item, ensure_ascii=False), allowed_names, first_ordinal + len(calls)
            )
            if call is not None:
                calls.append(call)
    return calls


def extract_tool_calls(text: str, allowed_names: set[str]) -> tuple[list[Any], str]:
    if not isinstance(text, str) or not text:
        return [], ""
    parser = _ToolBlockStream(allowed_names)
    for _ in parser.feed(text):
        pass
    for _ in parser.flush():
        pass
    return parser.calls, parser.text


def _token_count(usage: Any, name: str) -> int:
    """One integer token bucket of a usage record, or 0 when it is absent/unreadable."""
    try:
        return int(getattr(usage, name, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _prompt_size(usage: Any) -> int:
    """The prompt-token count of one usage record, or 0 when it is absent/unreadable."""
    return _token_count(usage, "input_tokens")


def _usage(result: Any) -> Any:
    usage = getattr(result, "usage", None)
    prompt = _token_count(usage, "input_tokens")
    completion = _token_count(usage, "output_tokens")
    cached = _token_count(usage, "cache_read_tokens")
    written = _token_count(usage, "cache_write_tokens")
    # ``prompt_tokens`` and friends stay the SDK's billed totals (summed over every turn and
    # retry); ``initial_*`` is the separate one-turn context anchor Hermes reads.
    anchor = {
        name: getattr(result, name)
        for name in ("initial_prompt_tokens", "initial_cached_tokens")
        if isinstance(getattr(result, name, None), int)
    }
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        prompt_tokens_details=SimpleNamespace(
            cached_tokens=cached,
            cache_write_tokens=written,
        ),
        **anchor,
    )


def _context_fields(turn_usage: Any) -> dict[str, int]:
    """Hermes' context-size anchor: ONE turn's prompt, separate from billed totals.

    Cursor's Python guide ("Token usage") is explicit: ``run.usage`` / ``result.usage`` hold a
    ``TokenUsage`` "summed across every turn that reported usage", while the ``usage`` stream
    event carries one turn's own numbers. Hermes reads ``usage.prompt_tokens`` as billed input
    (session totals, cost) but ``usage.initial_prompt_tokens`` — when present — as the size of
    the request it just sent, which anchors compression and the session budget. Reporting the
    run sum in that slot inflated a session by its turn count: one observed run reported
    598,980 for a ~241,305-token prompt, anchored a 33-message session above its 500,000
    window, and failed every later turn closed as "too large to compress further".

    The PEAK, not the last: a run that reports several turns need not finish with its largest
    prompt (one live thread measured 14,191 then 8,686 on consecutive calls), and understating
    the context is the direction that lets a session overflow before compression fires.

    ``initial_cached_tokens`` is capped at the anchor because Hermes subtracts it from the
    prompt; a cache reading larger than the prompt would make the context input bucket negative.
    """
    prompt = _prompt_size(turn_usage)
    if prompt <= 0:
        return {}
    cached = _token_count(turn_usage, "cache_read_tokens")
    return {
        "initial_prompt_tokens": prompt,
        "initial_cached_tokens": min(max(cached, 0), prompt),
    }


def _aggregate_result(attempts: list[tuple[Any, Any]]) -> Any:
    """Sum billed usage across discarded recovery attempts, keeping the final model.

    Billing really is spent on every attempt, so input/output/cache tokens add up. The
    context anchor must not: it describes one prompt, and summing 240k+241k+242k would report
    723k for a session that never exceeded 242k — the failure the anchor exists to prevent.
    """
    result, _ = attempts[-1]
    anchor = max(
        (item for _, item in attempts if item is not None), key=_prompt_size, default=None
    )

    def total(name: str) -> int:
        return sum(_token_count(getattr(item, "usage", None), name) for item, _ in attempts)

    return SimpleNamespace(
        model=getattr(result, "model", None),
        usage=SimpleNamespace(
            input_tokens=total("input_tokens"),
            output_tokens=total("output_tokens"),
            cache_read_tokens=total("cache_read_tokens"),
            cache_write_tokens=total("cache_write_tokens"),
        ),
        **_context_fields(anchor),
    )


def _tool_delta(call: Any, index: int) -> Any:
    return SimpleNamespace(
        index=index,
        id=call.id,
        type="function",
        function=SimpleNamespace(name=call.function.name, arguments=call.function.arguments),
    )


def _unsent_tail(delivered: str, full: str) -> str:
    """Suffix of the terminal text not already streamed, so completed text never repeats."""
    if not full or not full.startswith(delivered):
        return ""
    return full[len(delivered):]


_EXHAUSTED = object()


def _next_or_stop(iterator: Any) -> Any:
    """Pull one chunk without leaking StopIteration across an await boundary."""
    try:
        return next(iterator)
    except StopIteration:
        return _EXHAUSTED



def _drain(execution: Any) -> Any:
    """Exhaust an execution generator and return its (result, calls, content)."""
    while True:
        try:
            next(execution)
        except StopIteration as stop:
            return stop.value


def _possible_open_suffix(value: str) -> int:
    """Length of the longest suffix of ``value`` that could still become '<tool_call>'."""
    opener = _ToolBlockStream._OPEN
    for width in range(min(len(value), len(opener) - 1), 0, -1):
        if value.endswith(opener[:width]):
            return width
    return 0


class _ToolBlockStream:
    """Incremental splitter for the prompt-injected <tool_call> protocol.

    Emits ordinary text as soon as it is safe and buffers only a possibly-incomplete
    tool block, so arbitrary stream splits never tear a validated call. Blocks that do
    not validate are emitted as ``rejected`` events and re-emitted verbatim as text,
    matching ``extract_tool_calls``."""

    _OPEN = "<tool_call>"
    _CLOSE = "</tool_call>"

    def __init__(self, allowed_names: set[str], *, quiet: bool = False) -> None:
        self._allowed = allowed_names
        self._quiet = quiet
        self._buffer = ""
        self._text: list[str] = []
        self.calls: list[Any] = []
        self.rejected = 0
        self.duplicates = 0
        self._seen_calls: set[tuple[str, str]] = set()
        self._seen_ids: set[str] = set()
        self._scan = len(self._OPEN)
        self._quoted = self._escaped = False

    @property
    def text(self) -> str:
        return "".join(self._text).strip()

    def feed(self, chunk: str) -> Iterable[tuple[str, Any]]:
        self._buffer += chunk
        while True:
            start = self._buffer.find(self._OPEN)
            if start < 0:
                keep = _possible_open_suffix(self._buffer)
                if keep < len(self._buffer):
                    part, self._buffer = (
                        self._buffer[: len(self._buffer) - keep],
                        self._buffer[len(self._buffer) - keep:],
                    )
                    yield ("text", self._record(part))
                return
            if start > 0:
                part, self._buffer = self._buffer[:start], self._buffer[start:]
                yield ("text", self._record(part))
            end = self._block_end()
            if end < 0:
                return  # only an incomplete tool block stays buffered
            block, self._buffer = (
                self._buffer[: end + len(self._CLOSE)],
                self._buffer[end + len(self._CLOSE):],
            )
            self._scan = len(self._OPEN)
            self._quoted = self._escaped = False
            found = _tool_calls_from_block(
                block[len(self._OPEN): -len(self._CLOSE)].strip(),
                self._allowed,
                len(self.calls) + 1,
            )
            if not found:
                self.rejected += 1
                if not self._quiet:
                    _LOG.warning("cursor provider: rejected an unusable <tool_call> block (%d chars)", len(block))
                    _debug_event("rejected_tool_block", chars=len(block))
                yield ("rejected", self._record(block))
            else:
                for call in found:
                    if self._accept(call):
                        yield ("call", call)

    def _accept(self, call: Any) -> bool:
        # A model sometimes repeats the exact same call many times in one reply; executing every
        # copy repeats side effects (e.g. 20x kanban_request_review), so keep only the first.
        try:
            canonical = json.dumps(json.loads(call.function.arguments), sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            canonical = str(call.function.arguments)
        key = (call.function.name, canonical)
        if key in self._seen_calls:
            self.duplicates += 1
            _LOG.warning("cursor provider: dropped a duplicate %s call in the same reply", call.function.name)
            _debug_event("duplicate_tool_call", name=call.function.name)
            return False
        self._seen_calls.add(key)
        # Call ids must be unique within a reply so each result pairs with the right call.
        if call.id in self._seen_ids:
            base, n = call.id, 2
            while f"{base}_{n}" in self._seen_ids:
                n += 1
            call.id = call.call_id = f"{base}_{n}"
        self._seen_ids.add(call.id)
        self.calls.append(call)
        return True

    def _block_end(self) -> int:
        # Delimiters inside JSON strings are tool arguments, not protocol boundaries.
        while self._scan < len(self._buffer):
            char = self._buffer[self._scan]
            if not self._quoted and char == "<":
                if self._buffer.startswith(self._CLOSE, self._scan):
                    return self._scan
                if self._CLOSE.startswith(self._buffer[self._scan:]):
                    return -1
            if self._escaped:
                self._escaped = False
            elif self._quoted and char == "\\":
                self._escaped = True
            elif char == '"':
                self._quoted = not self._quoted
            self._scan += 1
        return -1

    def flush(self) -> Iterable[tuple[str, Any]]:
        if self._buffer:
            tail, self._buffer = self._buffer, ""
            # The buffer is being dropped, so its delimiter scan state is stale. The parser is
            # reused for a corrective retry; a fresh block must start scanning from the opener.
            self._scan = len(self._OPEN)
            self._quoted = self._escaped = False
            kind = "rejected" if tail.lstrip().startswith(self._OPEN) else "text"
            yield (kind, self._record(tail))

    def _record(self, part: str) -> str:
        if part:
            self._text.append(part)
        return part


def _native_tool_call(name: Any, args: Any, call_id: str, allowed: set[str], ordinal: int) -> Any | None:
    """Turn a Cursor-native tool_call event into a Hermes tool call when it names an offered tool."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if args is None:
        args = {}
    args = _plain(args)
    if not isinstance(args, dict):
        return None
    tool_name = str(name or "")
    if tool_name.replace("-", "_").lower() in _NATIVE_WRAPPERS:
        inner = args.get("toolName") or args.get("tool_name") or args.get("name")
        inner_args = args.get("arguments") or args.get("args") or {}
        if isinstance(inner_args, str):
            try:
                inner_args = json.loads(inner_args)
            except json.JSONDecodeError:
                return None
        if not inner or not isinstance(inner_args, dict):
            return None
        tool_name, args = str(inner), inner_args
    payload = {
        "id": call_id or f"cursor_native_{ordinal}",
        "function": {"name": tool_name, "arguments": json.dumps(args, ensure_ascii=False, separators=(",", ":"))},
    }
    return _tool_call_from_payload(json.dumps(payload), allowed, ordinal)


def render_correction(allowed: set[str], misses: list[str], rejected: int, *, native: bool = False) -> str:
    if native:
        parts = [
            "Your last reply did not produce a valid Hermes tool call. "
            "Call one of the offered tools through your tool-calling channel now."
        ]
        if misses:
            parts.append(
                "These native tool names do not exist in this runtime: " + ", ".join(sorted(set(misses))[:10]) + "."
            )
        if rejected:
            parts.append("A <tool_call> block was malformed or named an unknown tool.")
        parts.append("Allowed tool names: " + ", ".join(sorted(allowed)[:80]) + ".")
        return " ".join(parts)
    parts = ["Your last reply did not produce a valid Hermes tool call."]
    if misses:
        parts.append(
            "These native tool names do not exist in this runtime: " + ", ".join(sorted(set(misses))[:10]) + "."
        )
    if rejected:
        parts.append("A <tool_call> block was malformed or named an unknown tool.")
    parts.append(
        "Hermes tools are used ONLY by writing a <tool_call>{\"id\":\"call_1\",\"type\":\"function\",\"function\":"
        "{\"name\":\"NAME\",\"arguments\":\"{...}\"}}</tool_call> block as plain text. Redo the previous request now "
        "with a correct block. Allowed tool names: " + ", ".join(sorted(allowed)[:80]) + "."
    )
    return " ".join(parts)


def _split_model(model: str | None) -> tuple[str, list[dict[str, str]]]:
    """Split ``id@key=value,...`` into a Cursor model id and parameter list.

    Parameter ids/values are the ones Cursor's ``list_models()`` reports
    (e.g. ``reasoning_effort=xhigh,fast=true,context=500k``). Malformed
    suffixes fail instead of silently selecting the default variant."""
    value = str(model or "").strip()
    for prefix in ("cursor/", "cursor:"):
        if value.lower().startswith(prefix):
            value = value[len(prefix):]
            break
    base, sep, suffix = value.partition("@")
    base = base.strip()
    params: list[dict[str, str]] = []
    if sep:
        if not base:
            raise ValueError(f"Cursor model parameters need a model id: {value!r}")
        if "@" in suffix:
            raise ValueError(f"Cursor model parameters contain more than one '@': {value!r}")
        seen: set[str] = set()
        for item in suffix.split(","):
            key, eq, val = (part.strip() for part in item.partition("="))
            if not eq or not key or not val:
                raise ValueError(
                    f"Cursor model parameters must be key=value pairs: {value!r}"
                )
            if key in seen:
                raise ValueError(f"Cursor model parameters repeat {key!r}: {value!r}")
            seen.add(key)
            params.append({"id": key, "value": val})
    # Cursor requires an explicit model for local agents. ``auto`` is the
    # documented server-selected fallback; ``default`` is only our alias.
    if base.lower() in {"", "auto", "default"}:
        base = "auto"
    return base, params


def _plugin_params(model: str | None) -> tuple[list[dict[str, str]], bool | None]:
    """Cursor params plus the plugin-only ``native_tools`` override.

    ``on`` / ``off`` become True / False. Any other value is an error. The
    parameter is not a Cursor parameter and must not be sent to the SDK.
    """
    _, params = _split_model(model)
    cursor = [param for param in params if param["id"] != "native_tools"]
    chosen = next((param["value"] for param in params if param["id"] == "native_tools"), None)
    if chosen is None:
        return cursor, None
    if chosen not in {"on", "off"}:
        raise ValueError(
            f"Cursor model parameter 'native_tools' must be on or off; got {chosen!r}"
        )
    return cursor, chosen == "on"


def _model_id(model: str | None) -> str:
    """Normalised model name, including any parameters (used as a session key)."""
    base, params = _split_model(model)
    if not params:
        return base
    # Sort by id so `a=1,b=2` and `b=2,a=1` share one retained session.
    ordered = sorted(params, key=lambda param: param["id"])
    return base + "@" + ",".join(f"{p['id']}={p['value']}" for p in ordered)


def _model_selection(model: str | None) -> str | dict[str, Any]:
    """SDK ``model`` option. ``native_tools`` is stripped and never sent."""
    base, _ = _split_model(model)
    params, _ = _plugin_params(model)
    return {"id": base, "params": params} if params else base


def _model_parameter_values(models: Iterable[Any]) -> dict[str, dict[str, set[str]]]:
    """Model id -> parameter id -> accepted values from ``list_models()``."""
    catalog: dict[str, dict[str, set[str]]] = {}
    for model in models:
        model_id = str(getattr(model, "id", "") or "").strip()
        if not model_id:
            continue
        allowed: dict[str, set[str]] = {}
        for definition in getattr(model, "parameters", ()) or ():
            parameter_id = str(getattr(definition, "id", "") or "").strip()
            if not parameter_id:
                continue
            for item in getattr(definition, "values", ()) or ():
                value = str(getattr(item, "value", "") or "").strip()
                if value:
                    allowed.setdefault(parameter_id, set()).add(value)
        for variant in getattr(model, "variants", ()) or ():
            for item in getattr(variant, "params", ()) or ():
                parameter_id = str(getattr(item, "id", "") or "").strip()
                value = str(getattr(item, "value", "") or "").strip()
                if parameter_id and value:
                    allowed.setdefault(parameter_id, set()).add(value)
        catalog[model_id] = allowed
    return catalog


def _parameter_error(
    model_id: str, params: list[dict[str, str]], catalog: dict[str, dict[str, set[str]]]
) -> str | None:
    """Why *params* is unsupported for *model_id*, or None when it is valid."""
    allowed = catalog.get(model_id)
    if allowed is None:
        return f"Cursor model {model_id!r} is not in this account's model list"
    for param in params:
        values = allowed.get(param["id"])
        if values is None:
            supported = ", ".join(sorted(allowed)) or "none"
            return (
                f"Cursor model {model_id!r} has no parameter {param['id']!r} "
                f"(supported: {supported})"
            )
        if param["value"] not in values:
            choices = ", ".join(sorted(values)) or "none"
            return (
                f"Cursor model {model_id!r} parameter {param['id']!r} must be one of "
                f"{choices}; got {param['value']!r}"
            )
    return None


def _context_window(model: str | None) -> int | None:
    """Token window from a ``context=500k`` parameter, for Hermes' compression budget."""
    try:
        _, params = _split_model(model)
    except ValueError:
        return None
    for param in params:
        if param["id"] != "context":
            continue
        value = param["value"].strip().lower()
        multiplier = 1
        if value.endswith("k"):
            multiplier, value = 1_000, value[:-1]
        elif value.endswith("m"):
            multiplier, value = 1_000_000, value[:-1]
        try:
            tokens = int(float(value) * multiplier)
        except (ValueError, OverflowError):
            return None
        return tokens if tokens > 0 else None
    return None


def _assistant_history(content: str, calls: list[Any]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content or ""}
    canonical = _canonical_tool_calls(calls)
    if canonical:
        message["tool_calls"] = canonical
    return message


def _home_key() -> str:
    try:
        from hermes_constants import get_hermes_home

        home = Path(get_hermes_home())
    except (ImportError, RuntimeError):
        home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    return str(home.expanduser().resolve())


def _max_sessions() -> int:
    try:
        value = int(os.environ.get("HERMES_CURSOR_MAX_SESSIONS", "16"))
    except ValueError:
        value = 16
    return max(1, min(value, 128))


def _load_cursor_client() -> Any:
    try:
        return importlib.import_module("cursor_sdk").CursorClient
    except ImportError as exc:
        raise RuntimeError(
            "The Cursor provider requires cursor-sdk. Reinstall the plugin so Hermes installs its dependencies."
        ) from exc


@dataclass
class _Bridge:
    home: str
    workspace: tempfile.TemporaryDirectory[str]
    sdk: Any = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def client(self) -> Any:
        with self.lock:
            if self.sdk is None:
                self.sdk = _load_cursor_client().launch_bridge(
                    workspace=self.workspace.name,
                    state_root=str(Path(self.workspace.name) / "state"),
                    local={"cwd": self.workspace.name},
                    # Local bridge only: the SDK classifies run-scoped CancelRun/
                    # WaitLiveRun as cloud-routed, and its Client methods carry no
                    # apiKey, so with the guard on they fail before reaching this
                    # bridge. The flag's env fallback is not used by this plugin:
                    # CreateAgent/ListModels always receive the explicit account key,
                    # Send injects the agent's stored key, and a missing plugin key
                    # fails in _validate_request before any launch. Run-scoped calls
                    # carry no key; the local bridge resolves the live run created
                    # with it.
                    allow_api_key_env_fallback=True,
                )
            return self.sdk

    def close(self) -> None:
        with self.lock:
            sdk, self.sdk = self.sdk, None
        if sdk is not None:
            with suppress(Exception):
                sdk.close()
        self.workspace.cleanup()


@dataclass
class _Session:
    key: tuple[str, str, str, str]
    bridge: _Bridge
    cwd: Path
    agent: Any = None
    history: list[dict[str, Any]] = field(default_factory=list)
    tools_digest: str = ""
    retire_reason: str = "new"
    lock: threading.Lock = field(default_factory=threading.Lock)


_REGISTRY_LOCK = threading.RLock()
_BRIDGES: dict[str, _Bridge] = {}
_SESSIONS: OrderedDict[tuple[str, str, str, str], _Session] = OrderedDict()


def _new_bridge(home: str) -> _Bridge:
    root = Path(home) / "cache" / "cursor-sdk"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with suppress(OSError):
        root.chmod(0o700)
    workspace = tempfile.TemporaryDirectory(prefix="runtime-", dir=str(root))
    with suppress(OSError):
        Path(workspace.name).chmod(0o700)
    return _Bridge(home=home, workspace=workspace)


def _bridge_for(home: str) -> _Bridge:
    with _REGISTRY_LOCK:
        bridge = _BRIDGES.get(home)
        if bridge is None:
            bridge = _BRIDGES[home] = _new_bridge(home)
        return bridge


def _close_session(session: _Session) -> None:
    if session.agent is not None:
        _close_agent(session.agent)
        session.agent = None
    session.history.clear()
    shutil.rmtree(session.cwd, ignore_errors=True)


def _lease_session(key: tuple[str, str, str, str]) -> _Session | None:
    evicted: list[_Session] = []
    with _REGISTRY_LOCK:
        session = _SESSIONS.get(key)
        if session is None:
            bridge = _bridge_for(key[0])
            digest = hashlib.blake2b("\0".join(key).encode(), digest_size=16).hexdigest()
            sessions_root = Path(bridge.workspace.name) / "sessions"
            sessions_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            with suppress(OSError):
                sessions_root.chmod(0o700)
            # ponytail: unique paths keep delayed eviction cleanup from touching a replacement.
            cwd = Path(tempfile.mkdtemp(prefix=f"{digest[:8]}-", dir=str(sessions_root)))
            session = _SESSIONS[key] = _Session(key=key, bridge=bridge, cwd=cwd)
        if not session.lock.acquire(blocking=False):
            return None
        _SESSIONS.move_to_end(key)
        while len(_SESSIONS) > _max_sessions():
            victim_key = next(
                (
                    candidate_key
                    for candidate_key, candidate in _SESSIONS.items()
                    if candidate is not session and not candidate.lock.locked()
                ),
                None,
            )
            if victim_key is None:
                break
            evicted.append(_SESSIONS.pop(victim_key))
    for victim in evicted:
        _close_session(victim)
    return session


def _release_session(session: _Session) -> None:
    with _REGISTRY_LOCK:
        if _SESSIONS.get(session.key) is session:
            _SESSIONS.move_to_end(session.key)
        session.lock.release()


def _reset_session_registry() -> None:
    """Close process-owned Cursor resources. Public only for tests and atexit."""
    with _REGISTRY_LOCK:
        sessions = list(_SESSIONS.values())
        bridges = list(_BRIDGES.values())
        _SESSIONS.clear()
        _BRIDGES.clear()
    for session in sessions:
        _close_session(session)
    for bridge in bridges:
        bridge.close()


atexit.register(_reset_session_registry)


class _StreamControl:
    """Cross-thread handle so ``close()`` can cancel the run a worker is blocked on."""

    run: Any = None
    closed: bool = False


class _LazyStream:
    """One Cursor SDK run exposed as a lazy sync/async chunk stream.

    Nothing starts until the first pull; a pull drives the (blocking) SDK work on
    the calling thread, which for Hermes' Relay is its off-loop worker."""

    def __init__(self, client: "CursorSDKClient", kwargs: dict[str, Any]) -> None:
        self._client = client
        self._kwargs = kwargs
        self._closed = False
        self._control = _StreamControl()
        self._iterator: Any = None

    def _start(self) -> Any:
        if self._iterator is None:
            self._iterator = self._client._create_chat_completion(
                **self._kwargs, _cursor_stream_control=self._control
            )
            if self._closed:
                self._iterator.close()
        return self._iterator

    def __iter__(self):
        # ponytail: Relay calls iter() on-loop; its worker pulls next() off-loop.
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        try:
            return next(self._start())
        finally:
            if self._closed and self._iterator is not None:
                self._iterator.close()

    def __await__(self):
        # Resolve to an async iterable immediately; the run starts on the first pull.
        async def resolve() -> "_LazyStream":
            return self

        return resolve().__await__()

    async def __aiter__(self):
        try:
            while True:
                chunk = await asyncio.to_thread(_next_or_stop, self)
                if chunk is _EXHAUSTED:
                    return
                yield chunk
        finally:
            await asyncio.to_thread(self.close)

    def close(self) -> None:
        self._closed = True
        self._control.closed = True
        if self._iterator is None:
            return
        try:
            self._iterator.close()
        except ValueError:
            # A worker thread is inside next(); cancel its run so it unwinds there.
            _cancel_run(self._control.run)

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)


class _Completions:
    def __init__(self, client: "CursorSDKClient") -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return _LazyStream(self._client, kwargs)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self._client._create_chat_completion(**kwargs)
        return asyncio.to_thread(self._client._create_chat_completion, **kwargs)


class CursorSDKClient:
    """Hermes client seam backed only by Cursor's official SDK bridge."""

    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, *, api_key: Any = None, base_url: str | None = None, **_: Any) -> None:
        self.api_key = _secret(api_key)
        self._supplied_url = str(base_url or CURSOR_API_URL).rstrip("/")
        self.base_url = CURSOR_API_URL
        self.chat = SimpleNamespace(completions=_Completions(self))
        self.is_closed = False
        self._runs: set[Any] = set()
        self._cancelled_runs: set[Any] = set()
        self._runs_lock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _validate_request(self) -> None:
        if self.is_closed:
            raise RuntimeError("Cursor SDK client is closed")
        if self._supplied_url != CURSOR_API_URL:
            raise ValueError(f"Cursor SDK provider only accepts {CURSOR_API_URL}")
        if not self.api_key:
            raise RuntimeError(
                "CURSOR_API_KEY is not configured. Create a user API key at https://cursor.com/dashboard/api, "
                "then run `hermes auth add cursor`."
            )

    def _sdk_client(self) -> Any:
        self._validate_request()
        return _bridge_for(_home_key()).client()

    def cancel(self) -> None:
        with self._runs_lock:
            runs = tuple(self._runs)
            self._cancelled_runs.update(runs)
        for run in runs:
            _cancel_run(run)

    def close(self) -> None:
        if self.is_closed:
            return
        self.is_closed = True
        self.cancel()

    def _create_agent(
        self,
        *,
        bridge: _Bridge,
        cwd: Path,
        model: str | None,
        timeout: float,
        specs: list[dict[str, Any]] | None = None,
        scope: str = "ephemeral",
        reason: str = "new",
    ) -> Any:
        setup_client = bridge.client().with_options(
            timeout=timeout,
            # Keep the SDK's compact unary bound (30 s cap) for setup and for the
            # agent's run-scoped CancelRun/WaitLiveRun/CloseAgent calls. The plugin's
            # external deadline bounds how long it waits; stretching the transport's
            # own timeout would only retain abandoned cleanup workers longer.
            unary_timeout=min(timeout, 30.0),
            stream_timeout=timeout,
        )
        options: dict[str, Any] = {
            "api_key": self.api_key,
            "local": {"cwd": str(cwd)},
            "mcp_servers": {},
            "tools": [],
        }
        selected_model = _model_selection(model)
        if isinstance(selected_model, dict):
            catalog = _model_parameter_values(setup_client.list_models(api_key=self.api_key))
            error = _parameter_error(
                selected_model["id"], selected_model["params"], catalog
            )
            if error:
                raise ValueError(error)
        if selected_model:
            options["model"] = selected_model
        offered = list(specs or [])
        _LOG.info("cursor: agent created native_tools=%s tools=%d scope=%s reason=%s", "on" if offered else "off", len(offered), scope, reason)
        if offered:
            options["local"]["custom_tools"] = {
                spec["name"]: {"description": spec["description"], "input_schema": spec["parameters"], "execute": _inert_execute}
                for spec in offered
            }
            options["tools"] = ["mcp"]
        return setup_client.create_agent(options)

    def _consume(
        self,
        run: Any,
        parser: "_ToolBlockStream",
        allowed_names: set[str],
        native: dict[str, Any],
        *,
        deadline: float,
        cap: float,
        started: float,
        native_on: bool = False,
        control: Any = None,
    ) -> Any:
        """Stream one SDK run into ``parser``; returns (result, streamed, failure, stopped, peak).

        ``result.usage`` is left as the SDK's billed total; ``peak`` is this run's largest
        per-turn usage, which becomes the separate context anchor (see ``_context_fields``).
        """
        global _NATIVE_INTERCEPT_FORCED_LOGGED
        if native_on and not _flag("HERMES_CURSOR_INTERCEPT_NATIVE", True) and not _NATIVE_INTERCEPT_FORCED_LOGGED:
            _NATIVE_INTERCEPT_FORCED_LOGGED = True
            _LOG.info("cursor: interception forced on")
        intercept = True if native_on else _flag("HERMES_CURSOR_INTERCEPT_NATIVE", True)
        streamed = ""
        failure = ""
        stopped = False
        stopping = False
        call_cap = _native_max_calls() if native_on else 0
        capped = False
        post: _ToolBlockStream | None = None
        last_tick: float | None = None
        # The LARGEST per-turn ``usage`` event of this run. ``result.usage`` is the SDK's
        # run-wide sum, so it can never stand in for one prompt (see _context_fields).
        peak_usage: Any = None
        turn_text, turn_calls, closed = "", 0, False
        billed = SimpleNamespace(input_tokens=0, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0)

        def _calls_back(label: str | None = None, *, who: str | None = None, usage: Any = None):
            usage = usage or peak_usage or SimpleNamespace(input_tokens=0, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0)
            n = len(native["intercepted"])
            if who:
                _LOG.info("cursor: run cancelled by %s after %d native calls", who, n)
            elif label:
                _LOG.warning("cursor: run ended %s after %d native calls; returning them", label, n)
            return SimpleNamespace(result="", usage=usage, model=None, status="FINISHED"), streamed, "", True, peak_usage

        def _at_cap() -> bool:
            return bool(native_on and call_cap and len(parser.calls) >= call_cap)

        def _keep(call: Any, source: str) -> bool:
            nonlocal capped, last_tick
            if _at_cap():
                if not capped:
                    capped = True
                    _LOG.info("cursor: cap reached")
                return False
            if not parser._accept(call):
                return False
            native["intercepted"].append(call)
            _LOG.info("cursor: tool_call source=%s name=%s", source, call.function.name)
            last_tick = last_tick if last_tick is not None else time.monotonic()
            if _at_cap() and not capped:
                capped = True
                _LOG.info("cursor: cap reached")
            return True

        def _absorb(chunk: str):
            nonlocal post
            if post is None:
                post = _ToolBlockStream(allowed_names, quiet=True)
            for kind, value in post.feed(chunk):
                if kind == "call" and _keep(value, "text"):
                    yield ("call", value)

        def _requested(run_obj: Any) -> str | None:
            if self.is_closed or (control is not None and control.closed):
                return "hermes"
            with self._runs_lock:
                return "hermes" if run_obj in self._cancelled_runs else None

        stream = getattr(run, "stream", None)
        if callable(stream):
            iterator = iter(cast(Iterable[Any], stream()))
            while True:
                try:
                    event = _bounded_call(
                        lambda: next(iterator),
                        deadline=deadline,
                        cap=cap,
                        started=started,
                        phase="reading the run stream",
                    )
                except StopIteration:
                    break
                except Exception as exc:
                    if not (native_on and native["intercepted"]):
                        raise
                    _cancel_run(run)
                    label = "timeout" if getattr(exc, "error_code", "") == "cursor_run_timeout" else "error"
                    return _calls_back(label)
                kind = str(getattr(event, "type", ""))
                if native_on and last_tick is not None and time.monotonic() - last_tick >= 30:
                    last_tick = time.monotonic()
                    yield ("tick", None)
                if stopping and kind != "tool_call" and not native_on:
                    # The intercepted batch is complete; cancel before nested work or text runs.
                    break
                if kind == "assistant":
                    if native["intercepted"] and not native_on:
                        stopped = True  # model moved on after a native call; Hermes owns the loop
                        break
                    payload = getattr(event, "message", None)
                    for block in getattr(payload, "content", ()) or ():
                        text = str(getattr(block, "text", "") or "")
                        if not text:
                            continue
                        if native_on and closed and text.strip():
                            _cancel_run(run)
                            n = len(native["intercepted"])
                            _LOG.info("cursor: run ended on second answer after %d native calls", n)
                            native["cut"] = True
                            if n:
                                return _calls_back(usage=billed)
                            return SimpleNamespace(result=streamed, usage=billed, model=None, status="FINISHED"), streamed, "", True, peak_usage
                        if native_on:
                            turn_text += text
                        streamed += text
                        if native_on and native["intercepted"]:
                            yield from _absorb(text)
                        else:
                            yield from parser.feed(text)
                    if native_on and capped:
                        _cancel_run(run)
                        _LOG.info("cursor: run cancelled by plugin after %d native calls", len(native["intercepted"]))
                        return _calls_back()
                elif kind == "status":
                    status = str(getattr(event, "status", "") or "").upper()
                    if status in {"ERROR", "CANCELLED", "EXPIRED"}:
                        if native_on and native["intercepted"]:
                            _cancel_run(run)
                            who = _requested(run) if status == "CANCELLED" else None
                            return _calls_back(None if who else status.lower(), who=who)
                        failure = str(getattr(event, "message", "") or status)
                elif kind == "usage":
                    # One event per turn that reported usage, carrying THAT turn's numbers
                    # (Cursor Python guide, "Token usage"). Keep the LARGEST: a run that
                    # reports several turns need not finish with its biggest prompt, and
                    # the thread's peak is the conservative reading for a compression anchor.
                    turn_usage = getattr(event, "usage", None)
                    if turn_usage is not None and _prompt_size(turn_usage) > _prompt_size(
                        peak_usage
                    ):
                        peak_usage = turn_usage
                    if native_on:
                        if turn_text.strip() and turn_calls == 0 and not native["pending"]:
                            closed = True
                        if turn_usage is not None:
                            for name in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
                                setattr(billed, name, getattr(billed, name) + _token_count(turn_usage, name))
                        turn_text = ""
                        turn_calls = 0
                elif kind == "tool_call":
                    if native_on:
                        closed = False
                        turn_calls += 1
                    name = getattr(event, "name", "")
                    status = str(getattr(event, "status", "") or "").lower()
                    args = getattr(event, "args", None)
                    call_id = str(getattr(event, "call_id", "") or "")
                    if native_on:
                        call_id = call_id.split("\n", 1)[0].strip()
                    _debug_event(
                        "tool_call", name=name, status=status, call_id=call_id,
                        arg_keys=sorted(str(key) for key in args) if isinstance(args, dict) else [],
                        has_result=getattr(event, "result", None) is not None,
                    )
                    finished = status in {"completed", "error"}
                    if not intercept:
                        continue
                    if not finished:
                        # Arguments may still be streaming; remember them, act on the final event.
                        if call_id:
                            native["pending"][call_id] = (name, args)
                        continue
                    if call_id and call_id in native["ids"]:
                        continue
                    pending = native["pending"].pop(call_id, None) if call_id else None
                    if args in (None, {}, "") and pending is not None:
                        args = pending[1]
                    call = _native_tool_call(name, args, call_id, allowed_names, len(parser.calls) + 1)
                    if call is not None:
                        native["ids"].add(call_id or call.id)
                        if native_on:
                            if _keep(call, "native"):
                                yield ("call", call)
                            if capped:
                                _cancel_run(run)
                                _LOG.info("cursor: run cancelled by plugin after %d native calls", len(native["intercepted"]))
                                return _calls_back()
                        elif parser._accept(call):
                            native["intercepted"].append(call)
                            stopping = True
                            _LOG.warning(
                                "cursor provider: intercepted native tool call %r -> Hermes tool %r",
                                name, call.function.name,
                            )
                            _LOG.info("cursor: tool_call source=native name=%s", call.function.name)
                            yield ("call", call)
                    elif allowed_names and name:
                        miss = _native_miss_name(name, args) if native_on else str(name)
                        native["misses"].append(miss)
                        _LOG.warning("cursor provider: model called unknown native tool %r", name)
        else:
            iter_text = getattr(run, "iter_text", None)
            if callable(iter_text):
                iterator = iter(cast(Iterable[str], iter_text()))
                while True:
                    try:
                        part = _bounded_call(
                            lambda: next(iterator),
                            deadline=deadline,
                            cap=cap,
                            started=started,
                            phase="reading the run stream",
                        )
                    except StopIteration:
                        break
                    text = str(part or "")
                    if text:
                        streamed += text
                        yield from parser.feed(text)
        if intercept and not capped:
            for call_id, (name, args) in list(native["pending"].items()):
                if call_id in native["ids"]:
                    continue
                call = _native_tool_call(name, args, call_id, allowed_names, len(parser.calls) + 1)
                if call is None:
                    continue
                native["ids"].add(call_id or call.id)
                if native_on:
                    if _keep(call, "native"):
                        yield ("call", call)
                elif parser._accept(call):
                    native["intercepted"].append(call)
                    _LOG.warning("cursor provider: intercepted unfinished native tool call %r at run end", name)
                    _LOG.info("cursor: tool_call source=native name=%s", call.function.name)
                    yield ("call", call)
            native["pending"].clear()
        if native_on and native["intercepted"]:
            try:
                result = _bounded_call(
                    run.wait, deadline=deadline, cap=cap, started=started,
                    phase="waiting for the run result", grace=_WAIT_COMPLETION_GRACE_SECONDS,
                )
            except Exception as exc:
                _cancel_run(run)
                label = "timeout" if getattr(exc, "error_code", "") == "cursor_run_timeout" else "error"
                return _calls_back(label)
            status_value = getattr(getattr(result, "status", ""), "value", getattr(result, "status", ""))
            status = str(status_value or "").upper()
            if status in {"ERROR", "CANCELLED", "EXPIRED"}:
                who = _requested(run) if status == "CANCELLED" else None
                if who:
                    _LOG.info("cursor: run cancelled by %s after %d native calls", who, len(native["intercepted"]))
                else:
                    _LOG.warning("cursor: run ended %s after %d native calls; returning them", status.lower(), len(native["intercepted"]))
            tail = _unsent_tail(streamed, str(getattr(result, "result", "") or ""))
            if tail:
                yield from _absorb(tail)
            if post is not None:
                # flush() yields only text or rejected, so it can produce no calls.
                list(post.flush())
            return result, streamed, "", True, peak_usage
        if stopping:
            # A terminal native call was accepted; the run is cancelled rather than waited on.
            stopped = True
        if stopped:
            _cancel_run(run)
            result = SimpleNamespace(result="", usage=None, model=None, status="INTERCEPTED")
            return result, streamed, "", True, peak_usage
        result = _bounded_call(
            run.wait,
            deadline=deadline,
            cap=cap,
            started=started,
            phase="waiting for the run result",
            grace=_WAIT_COMPLETION_GRACE_SECONDS,
        )
        return result, streamed, failure, False, peak_usage

    def _run_agent(
        self,
        agent: Any,
        *,
        prompt: str,
        images: list[dict[str, str]],
        allowed_names: set[str],
        require_call: bool = False,
        control: _StreamControl | None = None,
        info: dict[str, Any] | None = None,
        native_on: bool = False,
    ) -> Any:
        """Run one logical turn with bounded, shared corrective recovery.

        Each attempt is exactly one SDK run (``_run_agent_once``), so retries are never
        nested or multiplied. A turn is accepted when it produced a valid call or an
        ordinary answer. Recovery covers required/named tool choices, short action-only
        announcements, unknown native tool names, and rejected blocks; native/rejected
        evidence keeps PR1's one-retry bound, action/required recovery gets PR3's two.
        """
        require_call = bool(allowed_names) and require_call
        info = info if info is not None else {}
        # Each attempt carries (result, its peak per-turn usage) so billed totals can be summed
        # without summing the one-turn context anchor.
        attempts: list[tuple[Any, Any]] = []
        budget: int | None = None
        fail_on_exhaustion = False
        for attempt in range(3):
            attempt_info: dict[str, Any] = {}
            execution = self._run_agent_once(
                agent, prompt=prompt, images=images, allowed_names=allowed_names,
                control=control, info=attempt_info, native_on=native_on,
            )
            held: list[tuple[str, Any]] = []
            dropped: list[str] = []
            holding = bool(allowed_names)
            try:
                while True:
                    try:
                        event = next(execution)
                    except StopIteration as stop:
                        result, calls, content, context_usage = stop.value
                        attempts.append((result, context_usage))
                        break
                    if event[0] == "tick":
                        yield event
                        continue
                    if holding:
                        held.append(event)
                        if event[0] == "call":
                            holding = False
                            # A rejected block in a turn that produced a valid call is
                            # discarded protocol, not answer text.
                            dropped.extend(
                                value for kind, value in held if kind == "rejected")
                            yield from (item for item in held if item[0] != "rejected")
                            held.clear()
                        elif not require_call and not _action_only("".join(
                            value for kind, value in held if kind == "text"
                        ), partial=True):
                            holding = False
                            yield from held
                            held.clear()
                    else:
                        yield event
            finally:
                execution.close()
            attempt_misses = list(attempt_info.get("misses", ()))
            info["native_misses"] = info.get("native_misses", 0) + attempt_info.get("native_misses", 0)
            info["intercepted"] = info.get("intercepted", 0) + attempt_info.get("intercepted", 0)
            info["rejected"] = info.get("rejected", 0) + attempt_info.get("rejected", 0)
            info["cut"] = info.get("cut") or attempt_info.get("cut")
            rejected = any(kind == "rejected" for kind, _ in held)
            held_text = "".join(value for kind, value in held if kind == "text")
            announced = bool(allowed_names) and _action_only(held_text or content)
            recovery = _flag("HERMES_CURSOR_RETRY", True) and bool(allowed_names) and (
                require_call or announced or rejected or bool(attempt_misses)
            )
            if calls or not recovery:
                yield from held
                for block in dropped:
                    content = content.replace(block, "", 1)
                return _aggregate_result(attempts), calls, content.strip()
            if budget is None:
                # Native/rejected evidence keeps the original single corrective retry;
                # required/named and announced actions get up to two nudges.
                fail_on_exhaustion = require_call or announced
                budget = 2 if fail_on_exhaustion else 1
            elif announced and budget < 2:
                # A retry that now announces an action inherits the action budget.
                fail_on_exhaustion = True
                budget = 2
            if attempt >= budget:
                if fail_on_exhaustion:
                    raise CursorSDKRunError(
                        "Cursor returned no valid Hermes tool call for a required or announced action; "
                        "recovery exhausted. No tool action was executed by the adapter."
                    )
                yield from held
                for block in dropped:
                    content = content.replace(block, "", 1)
                return _aggregate_result(attempts), calls, content.strip()
            info["retried"] = True
            # Once recovery starts, only a validated call can satisfy the promised action.
            require_call = True
            # One prompt covers both intents: name the native misses/rejections, list every
            # offered tool, and forbid another announcement or completion claim.
            prompt = render_correction(
                allowed_names, attempt_misses, attempt_info.get("rejected", 0), native=native_on,
            ) + " No tool was executed. Do not repeat an action announcement or claim completion. Hermes alone executes tools."
            _LOG.warning(
                "cursor provider: no valid tool call (native misses=%d, rejected blocks=%d); "
                "sending corrective retry %d/%d",
                len(attempt_misses), attempt_info.get("rejected", 0), attempt + 1, budget,
            )
            _debug_event("retry", misses=attempt_misses, rejected=attempt_info.get("rejected", 0))
            images = []

    def _run_agent_once(
        self,
        agent: Any,
        *,
        prompt: str,
        images: list[dict[str, str]],
        allowed_names: set[str],
        control: _StreamControl | None = None,
        info: dict[str, Any] | None = None,
        native_on: bool = False,
    ) -> Any:
        """Run exactly one SDK turn, yielding ('text'|'call', value) events as they stream.

        Native Cursor tool_call events that name an offered Hermes tool are converted into
        Hermes calls (HERMES_CURSOR_INTERCEPT_NATIVE, default on). ``info`` reports this
        attempt's native_misses / misses / intercepted / rejected so ``_run_agent`` can make
        one shared, bounded recovery decision; no retry happens here.

        Returns (result, calls, content, context_usage) — the extraction the buffered path
        used, produced incrementally so callers see text before the SDK run completes.
        ``context_usage`` is this run's peak per-turn usage (or None); ``result.usage`` keeps
        the SDK's billed run total."""
        parser = _ToolBlockStream(allowed_names)
        native: dict[str, Any] = {"ids": set(), "pending": {}, "intercepted": [], "misses": []}
        info = info if info is not None else {}
        # Start the clock before ``send()``: the SDK's Run constructor primes the Send
        # stream, so a stalled first byte is part of the run's blocking path too.
        cap = _run_timeout_seconds()
        started = time.monotonic()
        deadline = started + cap
        run = None
        try:
            run = _bounded_call(
                lambda: agent.send({"text": prompt, "images": images}),
                deadline=deadline,
                cap=cap,
                started=started,
                phase="starting the SDK run",
                late=_cancel_run,
            )
            with self._runs_lock:
                self._runs.add(run)
            if control is not None:
                control.run = run
                if control.closed or self.is_closed:
                    raise CursorSDKRunError("Cursor stream cancelled during startup")
            try:
                result, streamed, failure, stopped, context_usage = yield from self._consume(
                    run,
                    parser,
                    allowed_names,
                    native,
                    deadline=deadline,
                    cap=cap,
                    started=started,
                    native_on=native_on,
                    control=control,
                )
            finally:
                with self._runs_lock:
                    self._runs.discard(run)
                    self._cancelled_runs.discard(run)
            if not stopped:
                status_value = getattr(getattr(result, "status", ""), "value", getattr(result, "status", ""))
                status = str(status_value or "").upper()
                if failure or status in {"ERROR", "CANCELLED", "EXPIRED"}:
                    raise CursorSDKRunError(failure or f"Cursor SDK run ended with status {status}")
                tail = _unsent_tail(streamed, str(getattr(result, "result", "") or ""))
                if tail:
                    yield from parser.feed(tail)
            yield from parser.flush()
            info["native_misses"] = len(native["misses"])
            info["misses"] = list(native["misses"])
            info["intercepted"] = len(native["intercepted"])
            info["rejected"] = parser.rejected
            if native.get("cut"):
                info["cut"] = True
            return result, parser.calls, parser.text, context_usage
        except BaseException:
            _cancel_run(run)
            raise
        finally:
            if control is not None:
                control.run = None
            if run is not None:
                with self._runs_lock:
                    self._runs.discard(run)
                    self._cancelled_runs.discard(run)

    def _run_ephemeral(
        self,
        *,
        bridge: _Bridge,
        prompt: str,
        images: list[dict[str, str]],
        model: str | None,
        timeout: float,
        allowed_names: set[str],
        require_call: bool = False,
        control: _StreamControl | None = None,
        native_on: bool = False,
        specs: list[dict[str, Any]] | None = None,
    ) -> Any:
        sessions_root = Path(bridge.workspace.name) / "sessions"
        sessions_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.TemporaryDirectory(prefix="ephemeral-", dir=str(sessions_root)) as cwd:
            agent = self._create_agent(
                bridge=bridge, cwd=Path(cwd), model=model, timeout=timeout,
                specs=specs, scope="ephemeral", reason="new",
            )
            try:
                return (
                    yield from self._run_agent(
                        agent,
                        prompt=prompt,
                        images=images,
                        allowed_names=allowed_names,
                        require_call=require_call,
                        control=control,
                        native_on=native_on,
                    )
                )
            finally:
                _close_agent(agent)

    def _execute(
        self,
        *,
        model: str | None,
        messages: list[Any],
        tools: list[Any] | None,
        tool_choice: Any,
        timeout: float,
        session_scope: str,
        control: _StreamControl | None = None,
        stream: bool = False,
    ) -> Any:
        """Run one logical turn, yielding ('text'|'call', value) as the SDK streams.

        Returns (result, calls, content). Non-stream callers drain it; stream callers
        convert the events into OpenAI chunks before the SDK run completes."""
        self._validate_request()
        _, override = _plugin_params(model)  # reject a bad native_tools value before the bridge
        home = _home_key()
        bridge = _bridge_for(home)
        incoming = _conversation(messages)
        choice = _plain(tool_choice)
        require_call = choice == "required" or (
            isinstance(choice, dict) and isinstance(choice.get("function"), dict)
            and bool(choice["function"].get("name"))
        )
        specs = tool_specs(tools, tool_choice)
        switch = override if override is not None else _flag("HERMES_CURSOR_NATIVE_TOOLS", False)
        native_on = switch and bool(specs) and stream
        offered = specs if native_on else None

        if not session_scope:
            prompt, allowed_names = render_prompt(
                messages, tools=tools, tool_choice=tool_choice, native=native_on,
            )
            return (
                yield from self._run_ephemeral(
                    bridge=bridge,
                    prompt=prompt,
                    images=extract_images(messages),
                    model=model,
                    timeout=timeout,
                    allowed_names=allowed_names,
                    require_call=require_call,
                    control=control,
                    native_on=native_on,
                    specs=offered,
                )
            )

        account = hashlib.blake2b(self.api_key.encode(), digest_size=16).hexdigest()
        key = (home, session_scope, account, _model_id(model))
        session = _lease_session(key)
        if session is None:
            # ponytail: concurrent turns bypass shared state instead of queueing or corrupting it.
            prompt, allowed_names = render_prompt(
                messages, tools=tools, tool_choice=tool_choice, native=native_on,
            )
            return (
                yield from self._run_ephemeral(
                    bridge=bridge,
                    prompt=prompt,
                    images=extract_images(messages),
                    model=model,
                    timeout=timeout,
                    allowed_names=allowed_names,
                    require_call=require_call,
                    control=control,
                    native_on=native_on,
                    specs=offered,
                )
            )

        try:
            want = _tools_digest(specs) if native_on else ""
            if session.agent is not None and session.tools_digest != want:
                session.retire_reason = "toolset" if session.tools_digest and want else "mode"
                _close_agent(session.agent)
                session.agent = None
                session.history.clear()
            extends = (
                session.agent is not None
                and len(incoming) == len(messages)
                and len(incoming) > len(session.history)
                and incoming[: len(session.history)] == session.history
            )
            if session.agent is not None and not extends:
                session.retire_reason = "history"
                _close_agent(session.agent)
                session.agent = None
                session.history.clear()

            offset = len(session.history) if extends else 0
            request_messages = messages[offset:]
            if session.agent is None:
                reason = session.retire_reason
                session.agent = self._create_agent(
                    bridge=bridge,
                    cwd=session.cwd,
                    model=model,
                    timeout=timeout,
                    specs=offered,
                    scope="session",
                    reason=reason,
                )
                session.tools_digest = want
                session.retire_reason = "new"
            prompt, allowed_names = render_prompt(
                request_messages, tools=tools, tool_choice=tool_choice, native=native_on,
            )
            info: dict[str, Any] = {}
            result, calls, content = yield from self._run_agent(
                session.agent,
                prompt=prompt,
                images=extract_images(request_messages),
                allowed_names=allowed_names,
                require_call=require_call,
                control=control,
                info=info,
                native_on=native_on,
            )
            if (
                not calls
                and (info.get("native_misses") or info.get("retried"))
                and _flag("HERMES_CURSOR_RESET_POISONED", True)
            ):
                # The retained agent now remembers failed native-tool rounds; start clean next turn.
                _LOG.warning("cursor provider: resetting retained SDK agent after a failed tool-call turn")
                session.retire_reason = "poison"
                _close_agent(session.agent)
                session.agent = None
                session.history.clear()
                return result, calls, content
            if info.get("cut"):
                session.retire_reason = "cut"
                _close_agent(session.agent)
                session.agent = None
                session.history.clear()
                return result, calls, content
            session.history = incoming + [_assistant_history(content, calls)]
            return result, calls, content
        except BaseException:
            if session.agent is not None:
                _close_agent(session.agent)
            session.agent = None
            session.history.clear()
            session.retire_reason = "new"
            raise
        finally:
            _release_session(session)

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[Any] | None = None,
        tools: list[Any] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        timeout: Any = None,
        _cursor_session_scope: str | None = None,
        _cursor_stream_control: Any = None,
        **_: Any,
    ) -> Any:
        if stream:
            return self._stream_chat_completion(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                timeout=timeout,
                _cursor_session_scope=_cursor_session_scope,
                _cursor_stream_control=_cursor_stream_control,
            )
        result, calls, content = _drain(self._execute(
            model=model, messages=messages or [], tools=tools, tool_choice=tool_choice,
            timeout=_timeout_seconds(timeout), session_scope=str(_cursor_session_scope or "").strip(),
            stream=False,
        ))
        completion = SimpleNamespace(
            id=f"chatcmpl-cursor-{uuid.uuid4().hex}",
            object="chat.completion",
            model=str(model or getattr(getattr(result, "model", None), "id", "") or "cursor"),
            choices=[
                SimpleNamespace(
                    index=0,
                    message=SimpleNamespace(
                        role="assistant",
                        content=content or None,
                        tool_calls=calls or None,
                        reasoning=None,
                        reasoning_content=None,
                        reasoning_details=None,
                    ),
                    finish_reason="tool_calls" if calls else "stop",
                )
            ],
            usage=_usage(result),
        )
        return completion

    def _stream_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[Any] | None = None,
        tools: list[Any] | None = None,
        tool_choice: Any = None,
        timeout: Any = None,
        _cursor_session_scope: str | None = None,
        _cursor_stream_control: Any = None,
        **_: Any,
    ) -> Any:
        """Yield OpenAI-shaped chunks while the underlying Cursor run is still in flight."""
        execution = self._execute(
            model=model,
            messages=messages or [],
            tools=tools,
            tool_choice=tool_choice,
            timeout=_timeout_seconds(timeout),
            session_scope=str(_cursor_session_scope or "").strip(),
            control=_cursor_stream_control,
            stream=True,
        )
        stream_id = f"chatcmpl-cursor-{uuid.uuid4().hex}"
        response_model = str(model or "cursor")
        first = True
        content_started = False
        pending_ws = ""
        calls: list[Any] = []

        def chunk(*, content: str | None = None, tool_calls: Any = None, finish_reason: str | None = None) -> Any:
            nonlocal first
            delta = SimpleNamespace(
                role="assistant" if first else None,
                content=content,
                tool_calls=tool_calls,
                reasoning=None,
                reasoning_content=None,
            )
            first = False
            return SimpleNamespace(
                id=stream_id,
                object="chat.completion.chunk",
                model=response_model,
                choices=[SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)],
                usage=None,
            )

        try:
            while True:
                try:
                    kind, value = next(execution)
                except StopIteration as stop:
                    result, _, _ = stop.value
                    break
                if kind == "call":
                    calls.append(value)
                    yield chunk(tool_calls=[_tool_delta(value, len(calls) - 1)])
                    continue
                if kind == "tick":
                    yield chunk(content=None)
                    continue
                if not value:
                    continue
                stripped = value.rstrip()
                if not stripped:
                    pending_ws += value
                    continue
                if content_started:
                    piece = pending_ws + stripped
                else:
                    piece = stripped.lstrip()
                pending_ws = value[len(stripped):]
                if piece:
                    content_started = True
                    yield chunk(content=piece)

            response_model = str(
                model or getattr(getattr(result, "model", None), "id", "") or "cursor"
            )
            yield chunk(finish_reason="tool_calls" if calls else "stop")
            yield SimpleNamespace(
                id=stream_id,
                object="chat.completion.chunk",
                model=response_model,
                choices=[],
                usage=_usage(result),
            )
        finally:
            execution.close()


def list_cursor_models(*, api_key: str, timeout: float = 8.0) -> list[str]:
    """List account-visible models through the official SDK bridge."""
    if not _secret(api_key):
        return []
    try:
        CursorClient = importlib.import_module("cursor_sdk").CursorClient
    except ImportError:
        logging.getLogger(__name__).warning(
            "Cursor model listing skipped: the 'cursor-sdk' package is not importable by this "
            "Python environment. Install 'cursor-sdk>=1.0.32,<2' into the environment Hermes runs with."
        )
        return []
    with tempfile.TemporaryDirectory(prefix="hermes-cursor-models-") as workspace:
        client = CursorClient.launch_bridge(
            workspace=workspace,
            state_root=str(Path(workspace) / "state"),
            timeout=timeout,
            client_timeout=timeout,
            allow_api_key_env_fallback=False,
        )
        try:
            catalog = list(client.list_models(api_key=api_key))
            models = [
                str(model.id).strip() for model in catalog if str(model.id).strip()
            ]
        finally:
            client.close()
    return models + _configured_variants(models, catalog)


def _configured_variants(models: list[str], catalog: Iterable[Any]) -> list[str]:
    """Opt-in parameter variants to list alongside the base models.

    ``HERMES_CURSOR_MODEL_VARIANTS`` is a ``;``-separated list such as
    ``grok-4.7@context=500k,reasoning_effort=xhigh,fast=true``. Listing them lets
    Hermes' model validation (e.g. ``/model``) accept the parameterised name.
    Malformed entries, bare ids, variants of models the account cannot see, and
    unsupported parameter ids/values are skipped.
    """
    raw = os.environ.get("HERMES_CURSOR_MODEL_VARIANTS", "")
    known = set(models)
    allowed = _model_parameter_values(catalog)
    out: list[str] = []
    for entry in raw.split(";"):
        if not entry.strip():
            continue
        try:
            base, params = _split_model(entry)
            stripped, _ = _plugin_params(entry)
        except ValueError:
            continue
        name = _model_id(entry)
        if (
            params
            and base in known
            and name
            and name not in out
            and _parameter_error(base, stripped, allowed) is None
        ):
            out.append(name)
    return out
