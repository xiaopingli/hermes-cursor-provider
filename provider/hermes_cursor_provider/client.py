"""OpenAI-compatible facade over Cursor's official Python SDK."""
from __future__ import annotations

import atexit
import asyncio
import base64
import hashlib
import importlib
import json
import logging
import os
import queue
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

logger = logging.getLogger(__name__)

CURSOR_API_URL = "https://api.cursor.com"
_MAX_IMAGES = 4
_MAX_IMAGE_BASE64_CHARS = 28_000_000
_DEFAULT_MAX_TURN_TOOL_CALLS = 64
_MAX_MAX_TURN_TOOL_CALLS = 1024
# Session-thread budget. The SDK agent keeps its own accumulated thread and re-submits it
# on every send, so a long session can submit several times the model's declared window
# (the harness has seen a 3.9M-token input on a 500k model). When the last call's reported
# input crosses this budget the agent is retired and a fresh one starts from the full
# transcript. HERMES_CURSOR_SESSION_TOKEN_BUDGET overrides it, and outranks the model's own
# ``context=`` parameter so it can always disable or retune the policy at runtime.
_ENV_SESSION_TOKEN_BUDGET = "HERMES_CURSOR_SESSION_TOKEN_BUDGET"
_DEFAULT_SESSION_TOKEN_FRACTION = 1.0
_STREAM_HEARTBEAT_SECONDS = 60.0
# Wall-clock cap for one Cursor SDK run. The SDK's send/stream/wait path has no bounded
# wait of its own; without this a stalled run holds the Hermes turn open indefinitely.
_DEFAULT_RUN_TIMEOUT_SECONDS = 1800.0
_MIN_RUN_TIMEOUT_SECONDS = 0.05
_MAX_RUN_TIMEOUT_SECONDS = 86_400.0


class CursorSDKRunError(RuntimeError):
    """A terminal Cursor SDK run failure with Hermes-readable status."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        lowered = message.lower()
        if status_code is not None:
            self.status_code = status_code
        else:
            self.status_code = 429 if "out of usage" in lowered or "increase limits" in lowered else 502
        self.error_code = error_code or "cursor_sdk_run_failed"
        if body is not None:
            # The classifier reads the code off the parsed body (agent/error_classifier.py),
            # so the structured copy must travel with the exception, not only the attribute.
            self.body = body
        # Diagnostic only — nothing reads this. If it must be observable, put it in the message.
        self.recorded_calls: list[Any] = []


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


def render_prompt(
    messages: list[Any],
    *,
    tools: list[Any] | None = None,
    tool_choice: Any = None,
    native_tools: bool = False,
) -> tuple[str, set[str]]:
    specs = tool_specs(tools, tool_choice)
    allowed = {spec["name"] for spec in specs}
    sections = [
        "You are the model inside Hermes Agent. Hermes is the only tool control plane.",
    ]
    if native_tools and specs:
        sections.extend(
            [
                "Cursor built-in tools and MCP servers are disabled. The Hermes tools listed "
                "below are offered as native tool calls: Hermes executes each call and returns "
                "the result. Never claim to have read, written, executed, browsed, or changed "
                "anything unless the transcript contains the corresponding Hermes tool result.",
                "When a tool is needed, call it directly with the native tool-call mechanism. "
                "Do not describe the call in prose. Stop after the tool request; Hermes "
                "executes it and continues the turn.",
                "Hermes tools:\n" + json.dumps(specs, ensure_ascii=False, separators=(",", ":")),
            ]
        )
    else:
        sections.extend(
            [
                "Cursor built-in tools and MCP servers are disabled. Never claim to have read, written, "
                "executed, browsed, or changed anything unless the transcript contains the corresponding Hermes tool result.",
            ]
        )
        if specs:
            # QUARANTINED — unreachable in production, kept deliberately.
            #
            # Every production caller passes native_tools=bool(specs) (three sites in _execute,
            # each one line below its render_prompt call), so this branch is entered only when
            # specs is empty — and then the `if specs:` guard skips it. The text-protocol block
            # below is therefore dead code in production: the model is offered real custom tools
            # instead (see the native_tools branch above).
            #
            # It stays because tests/test_client.py:277
            # (test_native_prompt_drops_the_text_protocol) pins it for the non-native path, and
            # that pin is a deliberate safety net: if a future caller ever renders a prompt
            # without native tools, the model must still be told how to request one rather than
            # silently getting no protocol at all. Do not delete this block without moving that
            # test's intent first.
            #
            # The <tool_call> PARSER is a different thing and is live: _merged_calls (client.py
            # :489) is reached from the parse path, and it remains the safety net for a model
            # that emits the block form anyway.
            sections.extend(
                [
                    "When a tool is needed, output only one or more blocks in this exact form: "
                    '<tool_call>{"id":"call_1","type":"function","function":{"name":"NAME",'
                    '"arguments":"{\\"key\\":\\"value\\"}"}}</tool_call>. '
                    "The function name must be listed below and arguments must be valid JSON. Stop after the tool request; Hermes executes it.",
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
    sections.append("Continue from the final transcript entry.")
    return "\n\n".join(sections), allowed


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
        return None
    name = str(function.get("name") or "").strip()
    if not name or name not in allowed_names:
        return None
    arguments = function.get("arguments", "{}")
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


def extract_tool_calls(text: str, allowed_names: set[str]) -> tuple[list[Any], str]:
    if not isinstance(text, str) or not text:
        return [], ""
    parser = _ToolBlockStream(allowed_names)
    for _ in parser.feed(text):
        pass
    for _ in parser.flush():
        pass
    return parser.calls, parser.text


def _log_session_id() -> str:
    """The current Hermes session id, so this plugin's lines correlate with the core's.

    ``get_session_env`` is the supported accessor: the ContextVar wins under concurrency and
    falls back to ``os.environ`` for CLI/cron. Degrading to ``-`` is deliberate — a missing id
    must never break a run, and this is called from a logging path.
    """
    try:
        from gateway.session_context import get_session_env

        value = get_session_env("HERMES_SESSION_ID", "")
    except Exception:  # noqa: BLE001 - the id is diagnostic only
        value = os.environ.get("HERMES_SESSION_ID", "")
    text = str(value or "").strip()
    return text or "-"


def _usage(result: Any) -> Any:
    usage = getattr(result, "usage", None)
    prompt = int(getattr(usage, "input_tokens", 0) or 0)
    completion = int(getattr(usage, "output_tokens", 0) or 0)
    cached = int(getattr(usage, "cache_read_tokens", 0) or 0)
    written = int(getattr(usage, "cache_write_tokens", 0) or 0)
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        prompt_tokens_details=SimpleNamespace(
            cached_tokens=cached,
            cache_write_tokens=written,
        ),
    )


def _tool_delta(call: Any, index: int) -> Any:
    return SimpleNamespace(
        index=index,
        id=call.id,
        type="function",
        function=SimpleNamespace(name=call.function.name, arguments=call.function.arguments),
    )


def _new_tool_holder() -> dict[str, Any]:
    """Per-turn state the custom-tool closures resolve at call time.

    The agent (and therefore the SDK's registered handlers) outlives a turn, so the
    closures must read the *live* dict instead of capturing a list or a name set.
    """
    return {
        "sink": [],
        "allowed": set(),
        "registered": set(),
        "active": False,
        # Run-scoped dedupe state, published by ``_run_once`` at the start of each run.
        "gen": 0,
        "event_ids": set(),
    }


def _normalise_call_id(raw: object) -> str:
    """Cursor ids embed a newline ('call-…-1\\nfc_…_0'); keep the first segment.

    Returns "" when Cursor omitted the id — the caller drops that entry (fail
    closed); minting a random id here would guarantee a call that can never match.
    """
    return str(raw or "").split("\n", 1)[0].strip()


def _call_object_from_entry(entry: dict[str, Any], ordinal: int) -> Any:
    """Turn one reconciled sink entry into the pinned Hermes tool-call object."""
    call_id = str(entry.get("tool_call_id") or "").strip() or f"cursor_call_{ordinal}"
    arguments = entry.get("arguments") or {}
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    return SimpleNamespace(
        id=call_id,
        call_id=call_id,
        type="function",
        function=SimpleNamespace(name=str(entry.get("name") or ""), arguments=arguments),
    )


# What the model reads mid-run. It cannot see a tool result inside the turn (Hermes runs
# queued calls after the run, by design), so the ack is its only in-turn signal.
_ACK_ACCEPTED = (
    "accepted by Hermes: this tool is queued and runs after this turn; its result "
    "arrives in the next message. Do not call it again with the same arguments."
)
_ACK_DUPLICATE = (
    "already accepted this turn \u2014 this exact call is queued once; do not repeat it. "
    "Its result arrives in the next message."
)
_ACK_REFUSED_LIMIT = (
    "refused: turn limit of {cap} tool calls reached; the calls already accepted will "
    "still return their results. State your final answer now."
)
_TOOL_DESCRIPTION_SUFFIX = (
    "\n\n(Hermes executes this tool after your turn; its result arrives in the next "
    "message. Do not repeat a call you have already made this turn.)"
)


def hermes_tools_as_custom_tools(specs: list[dict[str, Any]], holder: dict[str, Any]) -> dict[str, Any]:
    """Register Hermes tools as Cursor custom tools whose callbacks only queue calls.

    ``execute`` never runs a tool: it records ``{name, arguments, tool_call_id, gen}`` on
    the current turn's sink and acks. Hermes executes the tool (approvals, logging, the
    real runner) once the call comes back on the completion.

    Guard order: active -> allowed -> call-id -> dedupe -> cap -> append. A repeat of an
    already-accepted call in the same run collapses to a duplicate ack without consuming
    cap budget; an id-less repeat still takes the id refusal.
    """

    def execute_factory(name: str) -> Any:
        def execute(args: Any, context: Any) -> str:
            if not holder["active"]:  # outside this turn's send()…run-end window
                logger.info("cursor: callback outside the run window for %s", name)
                return "refused: no active turn"
            if name not in holder["allowed"]:  # the LIVE set, not a captured one
                logger.info("cursor: callback for a tool this turn did not offer: %s", name)
                return "refused: tool not offered this turn"
            cid = _normalise_call_id(getattr(context, "tool_call_id", None))
            if not cid:  # Cursor omitted the id -> Hermes could never see it
                logger.info("cursor: refusing tool call %s: Cursor sent no call id", name)
                return "refused: the call was not delivered (no id); state the final answer"
            args_dict = dict(args)
            if _dedupe_enabled() and holder["sink"]:
                # Run-scoped and reconciled: only an entry from THIS run whose id this run
                # has seen as a tool_call event (the set ``_merged_calls`` will deliver)
                # can suppress a repeat, so a dedupe never withholds a call that would run.
                reconciled = {
                    _normalise_call_id(value) for value in (holder.get("event_ids") or ())
                }
                if any(
                    e.get("gen") == holder.get("gen", 0)
                    and _normalise_call_id(e.get("tool_call_id")) in reconciled
                    and e["name"] == name
                    and e["arguments"] == args_dict
                    for e in holder["sink"]
                ):
                    logger.info("cursor: duplicate tool call %s collapsed this run", name)
                    return _ACK_DUPLICATE
            cap = _max_turn_tool_calls()
            if len(holder["sink"]) >= cap:
                logger.info("cursor: refusing tool call %s: turn limit reached", name)
                return _ACK_REFUSED_LIMIT.format(cap=cap)
            holder["sink"].append(
                {
                    "name": name,
                    "arguments": args_dict,
                    "tool_call_id": cid,
                    "gen": holder.get("gen", 0),
                }
            )
            return _ACK_ACCEPTED

        return execute

    return {
        spec["name"]: {
            "description": (spec.get("description") or "") + _TOOL_DESCRIPTION_SUFFIX,
            "input_schema": spec["parameters"],
            "execute": execute_factory(spec["name"]),
        }
        for spec in specs
    }


def _merged_calls(
    holder: dict[str, Any] | None,
    event_ids: set[str],
    parser_calls: list[Any],
) -> list[Any]:
    """This turn's calls: reconciled native ones, else the honoured parser fallback.

    Reconciliation against the run's own ``tool_call`` events is the only filter, so a
    callback that lands late (or after a later turn rebound the holder) can never become
    a fresh Hermes tool call.
    """
    if not holder:
        return list(parser_calls)
    seen = {_normalise_call_id(value) for value in event_ids}
    native: list[Any] = []
    for entry in list(holder.get("sink") or []):
        cid = _normalise_call_id(entry.get("tool_call_id"))
        if not cid or cid not in seen:
            logger.info(
                "cursor: dropping a tool callback outside this run (%r, id %r)",
                entry.get("name"),
                cid,
            )
            continue
        native.append(_call_object_from_entry(entry, len(native) + 1))
    return native or list(parser_calls)


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
    not validate are re-emitted verbatim, matching ``extract_tool_calls``."""

    _OPEN = "<tool_call>"
    _CLOSE = "</tool_call>"

    def __init__(self, allowed_names: set[str]) -> None:
        self._allowed = allowed_names
        self._buffer = ""
        self._text: list[str] = []
        self.calls: list[Any] = []
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
            call = _tool_call_from_payload(
                block[len(self._OPEN): -len(self._CLOSE)].strip(),
                self._allowed,
                len(self.calls) + 1,
            )
            if call is None:
                yield ("text", self._record(block))
            else:
                self.calls.append(call)
                yield ("call", call)

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
            yield ("text", self._record(tail))

    def _record(self, part: str) -> str:
        if part:
            self._text.append(part)
        return part


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


PLAN_NUDGE = (
    "Your previous reply announced what you would do but did not call a tool, so nothing "
    "was executed. If you intended to use a tool, call it now using the tools provided in "
    "this conversation. If the work is finished, reply with your final answer prefixed by "
    "`FINAL:`."
)

_PLAN_RE = re.compile(
    r"\b(?:(?:first|next|now|ok(?:ay)?|sure|alright)[,!.]?\s+)?"
    r"(?:i['\u2019]ll|i will|i am going to|i['\u2019]m going to|let me|let['\u2019]s)\b",
    re.IGNORECASE,
)

# A pleasantry is not a statement of intent when it is the whole reply; it never vetoes
# an announcement that merely ends politely.
_PLEASANTRY_RE = re.compile(
    r"\b(let me know|i'?ll be happy to|would you like)\b",
    re.IGNORECASE,
)

_FINAL_RE = re.compile(r"\A\s*FINAL:", re.IGNORECASE)


def _plan_nudge_enabled() -> bool:
    return os.environ.get("HERMES_CURSOR_PLAN_NUDGE", "1").strip().lower() not in {"0", "false", "no", "off"}


def _dedupe_enabled() -> bool:
    return os.environ.get("HERMES_CURSOR_DEDUPE", "1").strip().lower() not in {"0", "false", "no", "off"}


def _looks_like_plan(text: str | None) -> bool:
    """A short statement of intent (or a dangling colon) rather than a final answer.

    The search is unanchored: the motivating failure ("The plan is loaded. Next I'll
    check…") begins mid-sentence. This gate only decides whether to spend one nudge; the
    raise is decided by the ``FINAL:`` marker, so a false positive cannot hard-fail an
    answer that is otherwise complete.
    """
    value = str(text or "").strip()
    if not value or len(value) > 800:
        return False
    intent = _PLEASANTRY_RE.sub(" ", value)
    if not intent.strip(" \t\r\n.!?,;:-"):
        return False
    return bool(_PLAN_RE.search(intent)) or value.rstrip().endswith(":")


def _strip_final_marker(text: str | None) -> str:
    """Remove a leading ``FINAL:``; unmarked text is returned unchanged.

    Same rule as the acceptance check (a blank line cannot hide the marker), but anchored:
    both call sites pass text that never went through the prefix gate, so an unanchored
    strip would delete a mid-sentence ``FINAL:`` along with the whitespace before it.
    """
    value = str(text or "")
    if not _FINAL_RE.match(value):
        return value
    return _FINAL_RE.sub("", value, count=1).lstrip()


def _model_id(model: str | None) -> str | None:
    """Normalised model name, including any parameters (used as a session key)."""
    base, params = _split_model(model)
    if not params:
        return base
    return base + "@" + ",".join(f"{p['id']}={p['value']}" for p in params)


def _model_selection(model: str | None) -> str | dict[str, Any]:
    """Value for the SDK's ``model`` option: a bare id or ``{id, params}``."""
    base, params = _split_model(model)
    return {"id": base, "params": params} if params else base


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


def _parse_token_count(raw: str) -> int | None:
    """A token count written as ``250000``, ``250k`` or ``2m``; None when unreadable.

    The single mapping both the model's ``context=`` parameter and the env override use, so
    ``250k`` means 250,000 in both places (Hermes's own mapping) instead of 256,000 or a
    silently different value. Never raises: an unreadable count is a missing one.
    """
    text = str(raw or "").strip().lower()
    if not text:
        return None
    try:
        if text.endswith("k"):
            return int(float(text[:-1]) * 1000)
        if text.endswith("m"):
            return int(float(text[:-1]) * 1_000_000)
        return int(text)
    except ValueError:
        return None


def _parse_context_window(model: str | None) -> int | None:
    """The declared context window carried in a model string, or None when absent.

    ``context=500k`` means 500,000 — the same mapping Hermes itself uses for this variant
    (``config.yaml`` pairs that string with ``context_window: 500000``), so the budget and
    the declared window agree instead of differing by 2.4%. A bare number is taken as-is;
    anything non-numeric is treated as absent. Parsing never raises: a malformed parameter
    is a missing one, which the caller handles by leaving the policy off.
    """
    try:
        _, params = _split_model(model)
    except ValueError:
        return None
    for param in params:
        if param.get("id") != "context":
            continue
        return _parse_token_count(str(param.get("value") or ""))
    return None


def _session_token_budget(model: str | None) -> int | None:
    """The session-thread budget in tokens, or None when the policy is off.

    Resolution order, deliberately env first: ``HERMES_CURSOR_SESSION_TOKEN_BUDGET`` is the
    kill switch and outranks the model's own parameter, because the live and harness model
    strings always carry ``context=500k`` — an env var that could be shadowed by the model
    string could never act on the running configuration. Accepted values: a positive
    integer (that budget); ``off`` or ``0`` (policy off); anything else is treated as unset
    with one warning. With no env value the budget follows the model's declared window;
    with neither, the policy is off and the caller logs that once per session.
    """
    raw = os.environ.get(_ENV_SESSION_TOKEN_BUDGET, "").strip()
    if raw:
        lowered = raw.lower()
        if lowered in {"off", "false", "no", "none"}:
            return None
        # Accept the same k/m notation the model's own context= parameter uses, so an override
        # can be written the way the value reads beside it (250k, not 250000).
        scaled = _parse_token_count(lowered)
        if scaled is not None:
            if scaled <= 0:
                return None
            return scaled
        logger.warning(
            "cursor: ignoring malformed %s=%r (expected a token count such as 250000 or 250k, or off)",
            _ENV_SESSION_TOKEN_BUDGET,
            raw,
        )
    window = _parse_context_window(model)
    if window is None:
        return None
    return int(window * _DEFAULT_SESSION_TOKEN_FRACTION)


def _should_retire_session(last_input_tokens: int, budget: int | None) -> bool:
    """True when the last call's reported input has reached the budget.

    A budget of None means the policy is off. A non-positive reported value means "unknown"
    (no call has completed yet, or the SDK reported nothing) and never retires: a fresh
    session must not be retired on its first turn.
    """
    if budget is None or last_input_tokens <= 0:
        return False
    return last_input_tokens >= budget


def _max_turn_tool_calls() -> int:
    """Turn cap for queued tool calls; HERMES_CURSOR_MAX_TURN_TOOL_CALLS overrides it.

    Unset or blank gives the default. A malformed or non-positive value also gives the
    default, with a warning (the same rule as ``_run_timeout_seconds``). A valid value is
    clamped into ``[1, _MAX_MAX_TURN_TOOL_CALLS]``, with a warning when the clamp changes
    it. The cap cannot be switched off: it exists to stop a runaway turn.
    """
    raw = os.environ.get("HERMES_CURSOR_MAX_TURN_TOOL_CALLS", "").strip()
    if not raw:
        return _DEFAULT_MAX_TURN_TOOL_CALLS
    try:
        value = int(raw)
    except ValueError:
        logger.warning("cursor: ignoring malformed HERMES_CURSOR_MAX_TURN_TOOL_CALLS=%r", raw)
        return _DEFAULT_MAX_TURN_TOOL_CALLS
    if value <= 0:
        logger.warning("cursor: ignoring non-positive HERMES_CURSOR_MAX_TURN_TOOL_CALLS=%r", raw)
        return _DEFAULT_MAX_TURN_TOOL_CALLS
    clamped = min(value, _MAX_MAX_TURN_TOOL_CALLS)
    if clamped != value:
        logger.warning(
            "cursor: clamping HERMES_CURSOR_MAX_TURN_TOOL_CALLS=%r to %d",
            raw,
            clamped,
        )
    return clamped


def _run_timeout_seconds() -> float:
    """How long one SDK run may take before the plugin stops waiting, in seconds.

    Default 1800 s (30 min): the longest legitimate call seen in the acceptance harness
    was 11.6 min on a 3.9M-token input, so 30 min leaves ~2.5x headroom while still
    failing well before a 45-60 min run-kill. A malformed or non-positive value falls
    back to the default — the cap exists to stop the hang, so it cannot be switched off.
    """
    raw = os.environ.get("HERMES_CURSOR_RUN_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return _DEFAULT_RUN_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        logger.warning("cursor: ignoring malformed HERMES_CURSOR_RUN_TIMEOUT_SECONDS=%r", raw)
        return _DEFAULT_RUN_TIMEOUT_SECONDS
    if value <= 0:
        logger.warning("cursor: ignoring non-positive HERMES_CURSOR_RUN_TIMEOUT_SECONDS=%r", raw)
        return _DEFAULT_RUN_TIMEOUT_SECONDS
    return min(max(value, _MIN_RUN_TIMEOUT_SECONDS), _MAX_RUN_TIMEOUT_SECONDS)


def _run_timeout_error(cap: float, elapsed: float) -> CursorSDKRunError:
    """The loud end of a stalled run: a timeout Hermes can retry or fall back from."""
    return CursorSDKRunError(
        f"Cursor SDK run did not finish within {cap:g}s ({elapsed:g}s elapsed); Hermes "
        "stopped waiting instead of hanging the turn. Raise "
        "HERMES_CURSOR_RUN_TIMEOUT_SECONDS if this workload legitimately needs longer.",
        status_code=408,
        error_code="cursor_run_timeout",
        body={"error": {"code": "cursor_run_timeout"}},
    )


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
                    allow_api_key_env_fallback=False,
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
    lock: threading.Lock = field(default_factory=threading.Lock)
    tool_holder: dict[str, Any] = field(default_factory=_new_tool_holder)
    # Reported input tokens of the last completed call in this session; 0 means unknown.
    # Written on the success path only (a failed run has no usage to read), and read at the
    # top of the session branch to decide whether the SDK thread has outgrown its budget.
    last_input_tokens: int = 0
    # One log line per session when the policy is off for want of any window, so the
    # unmitigated configuration is visible without spamming every turn.
    budget_warned: bool = False


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
        with suppress(Exception):
            session.agent.close()
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


class _RunPump:
    """One Cursor SDK run on its own daemon thread, posting progress to a queue.

    ``Agent.send``, the event stream and ``run.wait()`` block for as long as Cursor
    takes and the SDK offers no bounded wait, so running them on the caller's thread
    leaves a turn with no way to stop waiting. The blocking work happens here instead
    and the caller reads ``queue`` against a wall-clock deadline; when it overruns, the
    caller raises after :meth:`cancel`. Trade-off: a worker wedged inside the SDK is
    abandoned (daemon thread) rather than joined — a bounded turn, and one lost thread
    only when the run was already pathological.
    """

    EVENT = "event"
    TEXT = "text"
    CALL_ID = "call_id"
    DONE = "done"
    ERROR = "error"

    def __init__(
        self,
        client: "CursorSDKClient",
        agent: Any,
        payload: dict[str, Any],
        control: _StreamControl | None,
    ) -> None:
        self._client = client
        self._agent = agent
        self._payload = payload
        self._control = control
        self.queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.run: Any = None
        self.abandoned = False
        self._thread = threading.Thread(
            target=self._work, name="cursor-sdk-run", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def cancel(self) -> None:
        """Stop waiting: cancel the run if it exists yet; the worker exits on its next step."""
        self.abandoned = True
        run = self.run
        if run is not None:
            with suppress(Exception):
                run.cancel()

    def _work(self) -> None:
        try:
            result, failure = self._collect()
        except BaseException as exc:  # noqa: BLE001 - the caller re-raises it
            self._settle()
            self.queue.put((self.ERROR, exc))
            return
        self._settle()
        self.queue.put((self.DONE, (result, failure)))

    def _collect(self) -> tuple[Any, str]:
        run = self._agent.send(self._payload)
        self.run = run
        with self._client._runs_lock:
            self._client._runs.add(run)
        if self._control is not None:
            self._control.run = run
        if self.abandoned or self._client.is_closed or (
            self._control is not None and self._control.closed
        ):
            raise CursorSDKRunError("Cursor stream cancelled during startup")
        failure = ""
        stream = getattr(run, "stream", None)
        if callable(stream):
            for event in cast(Iterable[Any], stream()):
                if self.abandoned:
                    raise CursorSDKRunError("Cursor SDK run abandoned after the wait deadline")
                kind = str(getattr(event, "type", ""))
                # A heartbeat point for the streaming builder, which is chunk-silent
                # while it holds a reply (thinking events included).
                self.queue.put((self.EVENT, None))
                if kind == "tool_call":
                    self.queue.put(
                        (self.CALL_ID, _normalise_call_id(getattr(event, "call_id", None)))
                    )
                elif kind == "assistant":
                    payload = getattr(event, "message", None)
                    for block in getattr(payload, "content", ()) or ():
                        text = str(getattr(block, "text", "") or "")
                        if text:
                            self.queue.put((self.TEXT, text))
                elif kind == "status":
                    status = str(getattr(event, "status", "") or "").upper()
                    if status in {"ERROR", "CANCELLED", "EXPIRED"}:
                        failure = str(getattr(event, "message", "") or status)
        else:
            iter_text = getattr(run, "iter_text", None)
            if callable(iter_text):
                for part in cast(Iterable[str], iter_text()):
                    if self.abandoned:
                        raise CursorSDKRunError("Cursor SDK run abandoned after the wait deadline")
                    self.queue.put((self.EVENT, None))
                    text = str(part or "")
                    if text:
                        self.queue.put((self.TEXT, text))
        result = run.wait()
        return result, failure

    def _settle(self) -> None:
        """Report the run out of flight *before* the caller sees the report.

        A caller reacting to ``DONE``/``ERROR`` must never observe the run still in
        ``client._runs`` or still reachable from ``control.run`` for cancellation.
        """
        run = self.run
        if run is not None:
            with self._client._runs_lock:
                self._client._runs.discard(run)
        if self._control is not None and self._control.run is run:
            self._control.run = None


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
            run = self._control.run
            if run is not None:
                with suppress(Exception):
                    run.cancel()

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
        for run in runs:
            with suppress(Exception):
                run.cancel()

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
        specs: list[dict[str, Any]],
        holder: dict[str, Any] | None = None,
    ) -> Any:
        client = bridge.client().with_options(
            timeout=timeout,
            unary_timeout=min(timeout, 30.0),
            stream_timeout=timeout,
        )
        local: dict[str, Any] = {"cwd": str(cwd)}
        custom_tools = (
            hermes_tools_as_custom_tools(specs, holder) if holder is not None and specs else {}
        )
        if custom_tools:
            # "mcp" is the only capability group that still offers custom tools; an empty
            # list offers nothing at all, custom tools included (probe-custom-tool.json).
            local["custom_tools"] = custom_tools
        if holder is not None:
            # The SDK Agent exposes no options to read back, so this is the only record of
            # the set the live agent was created with (the recreate rule compares it).
            holder["registered"] = {spec["name"] for spec in specs}
        options: dict[str, Any] = {
            "api_key": self.api_key,
            "local": local,
            "mcp_servers": {},
            "tools": ["mcp"] if custom_tools else [],
        }
        selected_model = _model_selection(model)
        if selected_model:
            options["model"] = selected_model
        return client.create_agent(options)

    def _run_agent(
        self,
        agent: Any,
        *,
        prompt: str,
        images: list[dict[str, str]],
        allowed_names: set[str],
        control: _StreamControl | None = None,
        holder: dict[str, Any] | None = None,
        mode: str = "ephemeral",
    ) -> Any:
        """Run one turn; if it ends on a plan with tools available, nudge once for the call.

        Some models (notably Grok at high effort) announce "I'll read the file..." and stop,
        expecting a native tool call that this prompt-protocol transport never sees. Without
        a nudge Hermes treats the announcement as the final answer and closes the task."""
        result, calls, content = yield from self._run_once(
            agent,
            prompt=prompt,
            images=images,
            allowed_names=allowed_names,
            control=control,
            holder=holder,
            mode=mode,
        )
        if calls or not allowed_names or not _plan_nudge_enabled() or not _looks_like_plan(content):
            return result, calls, content
        # The streaming builder resets its hold here: everything before this point was an
        # announcement, not the answer.
        yield ("reply", None)
        result, calls, more = yield from self._run_once(
            agent,
            prompt=PLAN_NUDGE,
            images=[],
            allowed_names=allowed_names,
            control=control,
            holder=holder,
            mode=mode,
        )
        if calls:
            # The turn continues with a tool call, but a FINAL: prefix the nudged reply
            # emitted anyway must not reach the delivered text or session history.
            return result, calls, "\n\n".join(
                part for part in (content, _strip_final_marker(more)) if part
            )
        if not _FINAL_RE.match(str(more or "")):
            # Two strikes: the nudge was answered with another announcement. Raise HERE,
            # before this turn is stored and before the streaming builder emits anything,
            # so Hermes cannot report the turn as completed or swap in another provider.
            raise CursorSDKRunError(
                "Cursor announced work twice without a tool call or a FINAL: answer"
                + (f": {more[:300]!r}" if more else ""),
                status_code=400,
                error_code="cursor_plan_only",
                body={"error": {"code": "cursor_plan_only"}},
            )
        logger.info("cursor: accepting the nudged reply with a FINAL marker (raw: %.300r)", more)
        # Strip the marker from the nudged reply BEFORE the join: the joined string would
        # otherwise carry it mid-text, and the model would imitate it next turn.
        return result, calls, "\n\n".join(
            part for part in (content, _strip_final_marker(more)) if part
        )

    def _run_once(
        self,
        agent: Any,
        *,
        prompt: str,
        images: list[dict[str, str]],
        allowed_names: set[str],
        control: _StreamControl | None = None,
        holder: dict[str, Any] | None = None,
        mode: str = "ephemeral",
    ) -> Any:
        """Run one agent turn, yielding ('text'|'call', value) events as they stream.

        Returns (result, calls, content) — the extraction the buffered path used,
        produced incrementally so callers see text before the SDK run completes.
        ``calls`` is the merged list: this run's reconciled native calls, or the
        honoured parser calls when none were delivered.

        The blocking SDK work runs on a ``_RunPump`` thread; this generator reads its
        queue against a wall-clock deadline, so a stalled run raises instead of
        holding the turn open.
        """
        parser = _ToolBlockStream(allowed_names)
        event_ids: set[str] = set()
        if holder is not None:
            # Publish this run's generation and its (live, mutated-in-place) reconciled id
            # set before the pump starts, so no callback of this run can precede them. The
            # nudge's second run gets its own generation, so an earlier run's straggler
            # can never shadow a legitimate repeat here.
            holder["gen"] = holder.get("gen", 0) + 1
            holder["event_ids"] = event_ids
        cap = _run_timeout_seconds()
        started = time.monotonic()
        deadline = started + cap
        pump = _RunPump(self, agent, {"text": prompt, "images": images}, control)
        pump.start()
        streamed = ""
        # Stream-health counters. Timestamps are taken where the caller consumes the queue,
        # which is where a stall is actually felt; nothing here sits on the token path and no
        # observer thread is added. One line is emitted at exit, in the finally below.
        first_at: float | None = None
        last_at: float | None = None
        max_gap = 0.0
        events = 0
        close = "unknown"
        finished: Any = None
        try:
            while True:
                try:
                    kind, value = pump.queue.get(
                        timeout=max(deadline - time.monotonic(), 0.0)
                    )
                except queue.Empty:
                    close = "timeout"
                    raise _run_timeout_error(cap, time.monotonic() - started) from None
                now = time.monotonic()
                events += 1
                if first_at is None:
                    first_at = now
                elif last_at is not None:
                    gap = now - last_at
                    if gap > max_gap:
                        max_gap = gap
                last_at = now
                if kind == _RunPump.EVENT:
                    # A heartbeat point for the streaming builder, which is chunk-silent
                    # while it holds a reply (thinking events included).
                    yield ("event", None)
                elif kind == _RunPump.CALL_ID:
                    event_ids.add(value)
                elif kind == _RunPump.TEXT:
                    streamed += value
                    yield from parser.feed(value)
                elif kind == _RunPump.DONE:
                    result, failure = value
                    finished = result
                    close = "done"
                    break
                else:  # _RunPump.ERROR: the SDK work raised on the worker thread
                    close = "sdk_error"
                    raise value
            status_value = getattr(getattr(result, "status", ""), "value", getattr(result, "status", ""))
            status = str(status_value or "").upper()
            if failure or status in {"ERROR", "CANCELLED", "EXPIRED"}:
                close = f"status_{status.lower() or 'error'}"
                raise CursorSDKRunError(failure or f"Cursor SDK run ended with status {status}")
            tail = _unsent_tail(streamed, str(getattr(result, "result", "") or ""))
            if tail:
                yield from parser.feed(tail)
            yield from parser.flush()
            close = "done"
            return result, _merged_calls(holder, event_ids, parser.calls), parser.text
        except GeneratorExit:
            # The consumer stopped reading (a cancelled turn): a close reason of its own, not
            # an error. BaseException handling below still cancels the worker.
            close = "consumer_closed"
            pump.cancel()
            raise
        except BaseException:
            if close == "unknown":
                close = "error"
            # Cancelling is how the worker is stopped: it may be blocked inside the SDK
            # with no other way to observe that nobody is reading its queue any more.
            pump.cancel()
            raise
        finally:
            # Logging only. Cancel deliberately stays on the abnormal paths: after a normal
            # completion ``_settle`` has already released the run, and the pump's own handle
            # still points at it, so cancelling here would cancel a finished run — exactly
            # what ``_settle``'s contract exists to prevent.
            self._log_stream_health(
                mode=mode,
                close=close,
                started=started,
                first_at=first_at,
                max_gap=max_gap,
                events=events,
                prompt=prompt,
                result=finished,
            )

    def _log_stream_health(
        self,
        *,
        mode: str,
        close: str,
        started: float,
        first_at: float | None,
        max_gap: float,
        events: int,
        prompt: str,
        result: Any,
    ) -> None:
        """One structured line per SDK run: the numbers an unattributed stall needs.

        Deliberately a single line emitted at exit. The counters are plain floats updated in the
        consume loop, so nothing sits on the token path and no observer thread is added; the
        whole body is guarded because a logging failure must never change a run's outcome.

        ``ttfe=-1.00s`` means the run produced no event at all before it closed — the shape that
        a silent hang leaves behind. ``in_tokens=0`` means the run never returned a result, so
        the provider reported no usage (a stall or a timeout), not that it used no tokens.
        """
        try:
            duration = time.monotonic() - started
            ttfe = (first_at - started) if first_at is not None else -1.0
            tokens = 0
            if result is not None:
                with suppress(Exception):
                    tokens = int(_usage(result).prompt_tokens or 0)
            logger.info(
                "cursor: stream health mode=%s close=%s events=%d ttfe=%.2fs gap=%.2fs "
                "dur=%.2fs in_tokens=%d prompt_chars=%d session=%s",
                mode,
                close,
                events,
                ttfe,
                max_gap,
                duration,
                tokens,
                len(prompt),
                _log_session_id(),
            )
        except Exception:  # noqa: BLE001 - diagnostics must never break a turn
            return

    def _run_ephemeral(
        self,
        *,
        bridge: _Bridge,
        prompt: str,
        images: list[dict[str, str]],
        model: str | None,
        timeout: float,
        allowed_names: set[str],
        specs: list[dict[str, Any]],
        control: _StreamControl | None = None,
    ) -> Any:
        sessions_root = Path(bridge.workspace.name) / "sessions"
        sessions_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.TemporaryDirectory(prefix="ephemeral-", dir=str(sessions_root)) as cwd:
            # A one-shot agent gets its own holder, which retires with the agent below.
            holder = _new_tool_holder()
            holder["sink"] = []
            holder["active"] = True
            holder["allowed"] = {spec["name"] for spec in specs}
            agent = None
            try:
                agent = self._create_agent(
                    bridge=bridge,
                    cwd=Path(cwd),
                    model=model,
                    timeout=timeout,
                    specs=specs,
                    holder=holder,
                )
                return (
                    yield from self._run_agent(
                        agent,
                        prompt=prompt,
                        images=images,
                        allowed_names=allowed_names,
                        control=control,
                        holder=holder,
                    )
                )
            finally:
                holder["active"] = False
                if agent is not None:
                    with suppress(Exception):
                        agent.close()

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
    ) -> Any:
        """Run one logical turn, yielding ('text'|'call', value) as the SDK streams.

        Returns (result, calls, content). Non-stream callers drain it; stream callers
        convert the events into OpenAI chunks before the SDK run completes."""
        self._validate_request()
        _split_model(model)  # reject malformed model parameters before launching the bridge
        specs = tool_specs(tools, tool_choice)
        home = _home_key()
        bridge = _bridge_for(home)
        incoming = _conversation(messages)

        if not session_scope:
            prompt, allowed_names = render_prompt(
                messages, tools=tools, tool_choice=tool_choice, native_tools=bool(specs)
            )
            return (
                yield from self._run_ephemeral(
                    bridge=bridge,
                    prompt=prompt,
                    images=extract_images(messages),
                    model=model,
                    timeout=timeout,
                    allowed_names=allowed_names,
                    specs=specs,
                    control=control,
                )
            )

        account = hashlib.blake2b(self.api_key.encode(), digest_size=16).hexdigest()
        key = (home, session_scope, account, str(_model_id(model)))
        session = _lease_session(key)
        if session is None:
            # ponytail: concurrent turns bypass shared state instead of queueing or corrupting it.
            prompt, allowed_names = render_prompt(
                messages, tools=tools, tool_choice=tool_choice, native_tools=bool(specs)
            )
            return (
                yield from self._run_ephemeral(
                    bridge=bridge,
                    prompt=prompt,
                    images=extract_images(messages),
                    model=model,
                    timeout=timeout,
                    allowed_names=allowed_names,
                    specs=specs,
                    control=control,
                )
            )

        holder = session.tool_holder
        try:
            # Per turn, in this order: a fresh sink, the run window, then the live name set.
            holder["sink"] = []
            holder["active"] = True
            holder["allowed"] = {spec["name"] for spec in specs}
            extends = (
                session.agent is not None
                and len(incoming) == len(messages)
                and len(incoming) > len(session.history)
                and incoming[: len(session.history)] == session.history
            )
            toolset_changed = (
                session.agent is not None and holder["allowed"] != holder["registered"]
            )
            # Session-thread budget: the SDK agent re-submits its whole thread each send, so
            # retire it once the last call's reported input reached the model's declared
            # window (or the env override). Retiring re-sends the full transcript, which is
            # what keeps context while bounding what Cursor is asked to re-read.
            budget = _session_token_budget(model)
            over_budget = _should_retire_session(session.last_input_tokens, budget)
            if budget is None and not session.budget_warned:
                session.budget_warned = True
                logger.info(
                    "cursor: session token budget off for this session — no %s and no "
                    "context= parameter on %r, so the SDK thread is unbounded",
                    _ENV_SESSION_TOKEN_BUDGET,
                    model,
                )
            if over_budget:
                logger.info(
                    "cursor: retiring session agent at %d reported input tokens (budget %s)",
                    session.last_input_tokens,
                    budget,
                )
            if session.agent is not None and (not extends or toolset_changed or over_budget):
                with suppress(Exception):
                    session.agent.close()
                session.agent = None
                session.history.clear()
                # Re-send the whole transcript to the fresh agent; a delta alone would
                # silently drop the session's context.
                extends = False

            offset = len(session.history) if extends else 0
            request_messages = messages[offset:]
            if session.agent is None:
                session.agent = self._create_agent(
                    bridge=bridge,
                    cwd=session.cwd,
                    model=model,
                    timeout=timeout,
                    specs=specs,
                    holder=holder,
                )
            prompt, allowed_names = render_prompt(
                request_messages, tools=tools, tool_choice=tool_choice, native_tools=bool(specs)
            )
            result, calls, content = yield from self._run_agent(
                session.agent,
                prompt=prompt,
                images=extract_images(request_messages),
                allowed_names=allowed_names,
                control=control,
                holder=holder,
                mode="persistent",
            )
            session.history = incoming + [_assistant_history(content, calls)]
            # Record what this call actually submitted, so the next turn can decide whether
            # the SDK thread has outgrown its budget. Success path only: when _run_agent
            # raises, `result` is unbound and CursorSDKRunError carries no usage.
            session.last_input_tokens = int(_usage(result).prompt_tokens or 0)
            return result, calls, content
        except BaseException as exc:
            if isinstance(exc, CursorSDKRunError):
                # Diagnostic only — nothing reads this; the message is the observable channel.
                exc.recorded_calls = list(holder["sink"])
            if session.agent is not None:
                with suppress(Exception):
                    session.agent.close()
            session.agent = None
            session.history.clear()
            raise
        finally:
            # Spans the main run and the nudge run; a callback after this is out of window.
            holder["active"] = False
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
        """Yield OpenAI-shaped chunks while the underlying Cursor run is still in flight.

        On a tool-enabled turn the reply text is HELD until the turn decides: Hermes turns
        a stream error raised after the first text delta into a partial stub, so the
        fail-closed raise (``_run_agent``) can only be loud if no content was sent. The
        hold is per reply: the reply-boundary event drops the announcement, then the
        nudged reply is held to run end unless it begins with ``FINAL:``.
        """
        execution = self._execute(
            model=model,
            messages=messages or [],
            tools=tools,
            tool_choice=tool_choice,
            timeout=_timeout_seconds(timeout),
            session_scope=str(_cursor_session_scope or "").strip(),
            control=_cursor_stream_control,
        )
        stream_id = f"chatcmpl-cursor-{uuid.uuid4().hex}"
        response_model = str(model or "cursor")
        first = True
        content_started = False
        pending_ws = ""
        calls: list[Any] = []
        native = bool(tool_specs(tools, tool_choice))
        hold: list[str] = []
        nudged = False
        last_chunk_at = time.monotonic()

        def chunk(*, content: str | None = None, tool_calls: Any = None, finish_reason: str | None = None) -> Any:
            nonlocal first, last_chunk_at
            delta = SimpleNamespace(
                role="assistant" if first else None,
                content=content,
                tool_calls=tool_calls,
                reasoning=None,
                reasoning_content=None,
            )
            first = False
            last_chunk_at = time.monotonic()
            return SimpleNamespace(
                id=stream_id,
                object="chat.completion.chunk",
                model=response_model,
                choices=[SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)],
                usage=None,
            )

        def emit_text(value: str):
            nonlocal content_started, pending_ws
            stripped = value.rstrip()
            if not stripped:
                pending_ws += value
                return
            if content_started:
                piece = pending_ws + stripped
            else:
                piece = stripped.lstrip()
            pending_ws = value[len(stripped):]
            if piece:
                content_started = True
                yield chunk(content=piece)

        def release_hold():
            if hold:
                buffered = "".join(hold)
                hold.clear()
                yield from emit_text(buffered)

        try:
            while True:
                try:
                    kind, value = next(execution)
                except StopIteration as stop:
                    result, calls, _ = stop.value
                    break
                if kind == "reply":
                    # The announcement is not the answer: drop its held text. (The sync
                    # path joins both only because it returns one summary string.)
                    hold.clear()
                    nudged = True
                    continue
                if kind == "event":
                    # Chunk-shaped keepalive; content=None does not count as a delta, so
                    # it cannot soften the raise.
                    if native and time.monotonic() - last_chunk_at > _STREAM_HEARTBEAT_SECONDS:
                        yield chunk(content=None)
                    continue
                if kind == "call":
                    # Buffered: the merged list `_run_once` returns is the only source of
                    # call deltas (a parser call it did not honour must not leak out).
                    if native and not nudged:
                        yield from release_hold()  # a call means this reply is not an announcement
                    continue
                if not value:
                    continue
                if not native:
                    yield from emit_text(value)
                    continue
                hold.append(value)
                if not nudged and len("".join(hold).strip()) > 800:
                    # Longer than the intent cap: by definition not an announcement.
                    yield from release_hold()

            response_model = str(
                model or getattr(getattr(result, "model", None), "id", "") or "cursor"
            )
            if native and not calls and nudged:
                # The nudged reply: deliver it only when it begins with FINAL:, and strip
                # the marker (same rule as the _run_agent check). Without the marker
                # _run_agent has already raised, so there is nothing to flush.
                buffered = "".join(hold)
                hold.clear()
                if _FINAL_RE.match(buffered):
                    yield from emit_text(_strip_final_marker(buffered))
            elif nudged:
                # A call means this reply continues the turn, but a FINAL: prefix it
                # emitted anyway must not reach the delivered text (same rule as the
                # sync join in _run_agent).
                buffered = _strip_final_marker("".join(hold))
                hold.clear()
                yield from emit_text(buffered)
            else:
                yield from release_hold()
            for index, call in enumerate(calls):
                yield chunk(tool_calls=[_tool_delta(call, index)])
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
            models = [str(model.id).strip() for model in client.list_models(api_key=api_key) if str(model.id).strip()]
        finally:
            client.close()
    return models + _configured_variants(models)


def _configured_variants(models: list[str]) -> list[str]:
    """Opt-in parameter variants to list alongside the base models.

    ``HERMES_CURSOR_MODEL_VARIANTS`` is a ``;``-separated list such as
    ``grok-4.7@context=500k,reasoning_effort=xhigh,fast=true``. Listing them lets
    Hermes' model validation (e.g. ``/model``) accept the parameterised name.
    Malformed entries, bare ids and variants of models the account cannot see
    are skipped; Cursor still validates parameter values at run time."""
    raw = os.environ.get("HERMES_CURSOR_MODEL_VARIANTS", "")
    known = set(models)
    out: list[str] = []
    for entry in raw.split(";"):
        if not entry.strip():
            continue
        try:
            base, params = _split_model(entry)
        except ValueError:
            continue
        name = _model_id(entry)
        if params and base in known and name and name not in out:
            out.append(name)
    return out
