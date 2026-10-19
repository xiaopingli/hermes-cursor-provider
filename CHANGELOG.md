# Changelog

## Unreleased

- Adds Cursor model parameter selection: `model@key=value,...` (e.g. `grok-4.7@reasoning_effort=xhigh,fast=true`) is sent to the SDK as `{id, params}`. Malformed suffixes fail before the bridge launches; parameter variants get separate retained sessions. Verified live: an xhigh+fast selection runs, and an invalid value is rejected by Cursor.
- Adds opt-in `HERMES_CURSOR_MODEL_VARIANTS` (`;`-separated) so chosen parameter variants appear in the model listing and pass Hermes' `/model` validation.
- Nudges once when a reply has tools available but ends on a plan ("I'll read the file…") with no `<tool_call>`, instead of returning the announcement as the final answer. Disable with `HERMES_CURSOR_PLAN_NUDGE=0`.
- Fixes the in-turn tool-call ack: the model now reads an explicit "accepted by Hermes … runs after this turn" instead of a bare `queued` (which it took for failure and answered by re-issuing the same calls), and each custom tool's description says Hermes runs it after the turn. A repeat of an already-accepted `(tool, arguments)` call within one run is collapsed to a duplicate ack and does not use up the turn cap. Adds two toggles: `HERMES_CURSOR_MAX_TURN_TOOL_CALLS` (default 64, was a fixed 16; clamped to 1–1024, a malformed or non-positive value falls back to 64 with a warning) and `HERMES_CURSOR_DEDUPE` (default on; `0`/`false`/`no`/`off` turns the collapse off). The limit refusal text changes to "refused: turn limit of N tool calls reached …".

- Bounds the SDK session thread: the agent re-submits its whole accumulated thread on every send, so a long session could ask Cursor to re-read several times the model's declared window (3.9M-token input recorded on a 500k model). The agent is now retired once the last call's reported input reaches the session budget and a fresh one gets the full transcript, so context is kept while the re-read is bounded. The budget follows the model string's own `context=` parameter (`500k` = 500,000); `HERMES_CURSOR_SESSION_TOKEN_BUDGET` overrides it and outranks the parameter, and `off`/`0` disables the policy. With neither present the policy is off and logs once per session.

- Logs stream health per SDK run: one line at exit carrying `mode`, `close`, `events`, `ttfe`, `gap`, `dur`, `in_tokens`, `prompt_chars`, and the `session` id. This makes a stall attributable from logs alone — previously the core's `stream_error_cleanup` closes carried no session id and a stalled run left no line of its own. Counters are updated in the consume loop (no observer thread, nothing on the token path) and the whole logging body is guarded, so diagnostics can never change a run's outcome. Adds `scripts/stream_health.py`, which summarises a log into a stream-health block and marks any session tag it could only infer.
- Documents that the persistent session path (retained agents, the token budget) requires a caller to pass `_cursor_session_scope`, which Hermes core does not do today; live turns take the ephemeral path.

## 0.3.6

- Replaces synthesized, post-completion chunks with genuine incremental SDK text streaming for Hermes Relay and async consumers. Awaiting a stream no longer collects the whole run.
- Parses tool-call blocks incrementally, including split delimiters and delimiter text inside JSON arguments; ordinary text is forwarded while Hermes retains tool execution and approvals.
- Preserves usage, session reuse, and terminal error propagation without duplicating the SDK's final text. Closing a stream cancels its run and releases session state, including cancellation during startup.
- Adds first-chunk-before-completion regressions and a real managed Relay gate test. Live Relay/async streaming and a Hermes tool round trip were verified.

## 0.3.5

- Fixes premature fallback under Hermes' managed Relay: streaming requests return a lazy sync/async-compatible stream instead of an unawaited coroutine.
- Keeps SDK work off the Relay event loop, executes each stream only once, and preserves async auxiliary calls, provider errors, and client cancellation.
- Adds regressions for Relay-style iteration, asynchronous streams, closing unconsumed streams, and cancellation; the current-main probe now exercises a real managed Relay stream.

## 0.3.4

- Reports Cursor SDK cache-write tokens to Hermes and maps Cursor's meter-sum `total_tokens` to the OpenAI-compatible prompt-plus-completion total.

## 0.3.3

- Surfaces terminal Cursor SDK failures (including exhausted model usage) as real provider errors instead of empty successful responses, so Hermes logs the Cursor reason before applying its configured fallback.

## 0.3.2

- Reads assistant text from the official SDK run stream before collecting the terminal result. Local SDK runs can leave `RunResult.result` empty even when they emitted a valid answer; that previously looked like an empty model response and triggered Hermes fallback.

## 0.3.1

- Stops requesting Cursor's optional local sandbox, which caused every SDK run to fail on unsupported hosts before Hermes fell back. Private per-agent working directories and the Hermes-owned tool boundary remain unchanged.

## 0.3.0

- Retains one official SDK agent per Hermes conversation/account/model and sends only appended transcript messages.
- Shares one warm SDK bridge per Hermes profile process and preserves retained agents across Hermes client rebuilds.
- Adds an idle-session LRU target, profile/account/model isolation, safe rewrite resets, concurrent-turn isolation, and process-exit cleanup without destructive SDK deletion.
- Documents Cursor cache semantics and exposes `HERMES_CURSOR_MAX_SESSIONS` (default 16).

## 0.2.1

- Always passes Cursor's documented `auto` model explicitly; local SDK agents reject an omitted model.

## 0.2.0

- Replaced the reverse-engineered Cursor Agent protocol with Cursor's official Python SDK and bundled bridge.
- Replaced browser-token OAuth with `CURSOR_API_KEY`, enabling official subscription-backed SDK usage and Hermes auxiliary routing.
- Removed vendored protobuf, HTTP/2 framing, private endpoints, token refresh code, and CLI impersonation.
- Added isolated per-agent workspaces, disabled Cursor built-in tools/MCP, allowlisted Hermes tool-call translation, vision inputs, official model discovery, and sync/async client support.
- Reworked documentation, security model, tests, and current-main verification for the SDK architecture.

## 0.1.0

- Initial standalone current-main Hermes model-provider plugin using Cursor's private Agent protocol.
