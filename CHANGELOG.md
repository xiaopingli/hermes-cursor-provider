# Changelog

## Unreleased

- Opt-in native Hermes tools behind `HERMES_CURSOR_NATIVE_TOOLS` (default off). On the stream path, offered Hermes tools are registered as inert Cursor custom tools in the `mcp` capability group; no MCP servers are loaded, and Hermes still decides and runs every tool. A call cap (`HERMES_CURSOR_NATIVE_MAX_CALLS`, default 64) and stream keep-alives bound the run. Non-stream requests stay on the upstream text protocol. Unset the switch to turn it off.
- Per-request `native_tools=on|off` in the model-name suffix overrides `HERMES_CURSOR_NATIVE_TOOLS` both ways and is not sent to Cursor. Under a multiplexing gateway, set the env form and `HERMES_CURSOR_MODEL_VARIANTS` in the launch `.env`; a named profile's `.env` does not reach the plugin.
- A native run is cancelled when Cursor starts a second answer. The session agent is retired and the next turn re-sends the full transcript. Usage is the sum of turns up to the cut, so Cursor's dashboard can show slightly more than Hermes reports.
- Bounds the SDK session thread with `HERMES_CURSOR_SESSION_TOKEN_BUDGET` (default: the model's `context=` window, `500k` = 500,000; the env value outranks the parameter; `off`/`0` disables). The agent is retired once the last call's reported input reaches the budget and a fresh one gets the full transcript (`reason=budget`). With neither the env var nor a `context=` parameter the policy is off and logs once per session.

## 0.4.1

- Keeps `usage.prompt_tokens`, `completion_tokens`, and the cache details as the SDK's **billed totals** (summed across every turn and every recovery attempt), and reports the run's **largest single turn** separately through `usage.initial_prompt_tokens` / `usage.initial_cached_tokens` — the hook Hermes reads as the size of the request it just sent. Replacing the whole usage object with the peak (as the previous revision did) dropped the other turns' output and cache spend from billing, and summing the anchor across recovery attempts reports 723k for three 240k/241k/242k-token prompts, failing a session that fits its window closed as "too large to compress further". The peak, not the last event: a run's turns are not ordered by size (one live thread measured 14,191 then 8,686 on consecutive calls). A run with no per-turn event keeps `result.usage` and reports no anchor. The context split requires Hermes >= 0.22.0; older hosts ignore the extra fields and anchor on the billed total.
- Bounds one Cursor SDK run end to end with `HERMES_CURSOR_RUN_TIMEOUT_SECONDS`: default `1800` (30 min), clamped to `0.05`–`86400`; malformed, non-finite, or non-positive values fall back to the default and the cap cannot be switched off. The deadline covers the blocking Send stream (including the SDK's priming read), every lazy stream pull, and the result wait. The SDK has no total-deadline option, so each blocking read runs on a daemon thread and is joined against the same deadline; exactly one read is in flight and only when the consumer asks, preserving incremental delivery and backpressure. A completed read hands its result to exactly one owner: the caller, or the timeout path's late cleanup (so a `send()` that finishes after the deadline still cancels the run it created). At the deadline a retryable `CursorSDKRunError` (status 408, `error_code=cursor_run_timeout`) is raised; Hermes classifies 408 as a timeout and retries it, and no fallback is promised. Run-scoped `CancelRun`/`WaitLiveRun` now reach the locally launched bridge; their SDK unary timeout stays at the compact 30 s cap, which also reclaims abandoned cleanup workers without stretching a healthy limit. Cancel/close are bounded to a short, cap-scaled grace so cleanup cannot add another transport timeout to the user-visible deadline. Remaining ceiling: a transport read already in flight cannot be interrupted; its worker is abandoned as a daemon thread and is reclaimed only when that call returns or times out (stream reads use the Hermes request timeout, 300 s by default; unary cleanup uses the SDK's 30 s bound), or when the server honours cancellation.

## 0.4.0

- Handles the SDK's native `tool_call` events. When the model calls an offered Hermes tool through Cursor's native channel (which the runtime answers with "Tool not found"), the call is converted into a Hermes tool call and the run is cancelled. Disable with `HERMES_CURSOR_INTERCEPT_NATIVE=0`.
- Matches model-written tool names against offered tools (prefixes like `functions.`/`mcp_`, CamelCase, unique multi-word suffix) and unwraps Cursor's `CallMcpTool` wrapper.
- Accepts `<tool_call>` JSON without the `function` wrapper (`name` + `arguments`/`parameters`/`args`/`input`).
- Stronger prompt rules: Hermes tools are not native, "Tool not found" means wrong channel, never claim tools are unavailable; the rule is repeated right before the final transcript entry.
- Adds bounded recovery for short action-only replies and unsatisfied required/named tool choices, without enabling Cursor-native tools. Recovery is shared and not nested: an unknown native tool or a rejected `<tool_call>` block gets one corrective retry, action-only or required/named turns get up to two nudges. Disable with `HERMES_CURSOR_RETRY=0`.
- Keeps substantive first-person answers that continue after a colon, semicolon, or spaced dash out of action-only recovery, and treats malformed or unoffered tool blocks after an announcement as rejected protocol instead of leaking raw XML as the final answer.
- Keeps buffered answers, streamed output, and retained history consistent when rejected protocol is withheld; reports usage summed across discarded recovery attempts, including cache reads and writes.
- Resets a retained SDK agent after a failed tool-call turn so later turns do not inherit "Tool not found" history. Disable with `HERMES_CURSOR_RESET_POISONED=0`.
- Native calls are only converted on a final (completed or error) event, using that event's arguments (pending args from earlier running events are the fallback); calls still unfinished when the run ends are converted then.
- Diagnostics: warnings through the `hermes_cursor_provider` logger; bounded event metadata only (never raw arguments, results, or rejected blocks) to `<HERMES_HOME>/logs/cursor-provider-events.jsonl` when `HERMES_CURSOR_DEBUG_EVENTS=1`. Log files use owner-only permissions. Adds `scripts/native_tool_probe.py`.
- Resets incomplete-block parser state before corrective retries, deduplicates native calls through the shared acceptance path, and forwards contiguous terminal-event siblings before cancelling the intercepted run.
- Adds Cursor model parameter selection: `model@key=value,...` (e.g. `grok-4.7@fast=true,reasoning_effort=xhigh`) is sent to the SDK as `{id, params}`. Malformed suffixes fail before the bridge launches; parameter ids and values are checked against the account's `list_models()` catalog before agent creation. Parameter variants get separate retained sessions, independent of parameter order, and `context=` sets Hermes' compression budget. Automatic mapping of Hermes `/reasoning` settings is not included.
- Adds opt-in `HERMES_CURSOR_MODEL_VARIANTS` (`;`-separated) so chosen parameter variants appear in the model listing and pass Hermes' `/model` validation; unsupported ids/values are skipped.
- Logs a warning when the model list cannot be loaded (for example, `cursor-sdk` is not installed in Hermes' Python environment) instead of silently showing only the seed models.
- README: troubleshooting for a model picker that shows only `auto` and `composer-2.5`.

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
