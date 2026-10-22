# hermes-cursor-provider

Standalone Cursor subscription provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent), built for the current model-provider plugin API with **zero Hermes core changes**.

It uses Cursor's official [`cursor-sdk`](https://cursor.com/docs/sdk/python) package and its bundled local SDK bridge. A Cursor user API key charges runs to the same Cursor plan/request pool as SDK usage in Cursor.

## What it does

- Registers `cursor` through `ProviderProfile.create_client()`.
- Uses `CURSOR_API_KEY` through Hermes' normal API-key credential flow.
- Lists account-visible models through the official SDK.
- Converts Hermes chat history, images, and tool schemas into SDK agent requests.
- Disables Cursor built-in tools and MCP servers, gives each SDK agent a private working directory, and accepts only tool names offered by Hermes.
- Returns OpenAI-shaped sync/async responses so Hermes retains its normal conversation, approval, and tool loop.

Cursor's SDK is an agent surface, not a raw chat-completions API. The plugin retains one SDK agent per Hermes conversation/account/model, sends only newly appended messages, and offers Hermes' tools to Cursor as native custom tools. Those callbacks never execute a tool: each queues the call and acks it, and Hermes runs it after the turn through its normal approval loop, so tool execution never escapes Hermes' approval path. The older `<tool_call>` text protocol is retained only for the path where no tool schemas are supplied; whenever specs are present — every production call — native custom tools are the transport.

## Install

```bash
hermes plugins install 'https://github.com/anpicasso/hermes-cursor-provider.git#provider' --enable
hermes auth add cursor
hermes model
```

> **This fork.** `xiaopingli/hermes-cursor-provider` is a public fork carrying work not yet in the
> upstream plugin: native custom tools with the queue/ack contract, the bounded session thread, and
> per-run stream-health logging. Install the fork instead of the stock plugin by swapping the URL:
>
> ```bash
> hermes plugins install 'https://github.com/xiaopingli/hermes-cursor-provider.git#provider' --enable
> ```

Create a **user API key** at <https://cursor.com/dashboard/api>. User and service-account API keys are supported by Cursor's SDK; Team Admin keys are not inference credentials.

Plugins and credentials are profile-scoped:

```bash
hermes --profile work plugins install 'https://github.com/anpicasso/hermes-cursor-provider.git#provider' --enable
hermes --profile work auth add cursor
```

Test without changing the configured default model:

```bash
hermes chat --provider cursor --model auto -q 'Reply with exactly: OK'
```

### Model parameters (effort, fast, context)

Cursor models expose parameters such as reasoning effort, fast mode and context size (see each model's `parameters` in the SDK's `list_models()`). Append them to the model name as `@key=value,...`:

```bash
hermes chat --provider cursor --model 'grok-4.7@reasoning_effort=xhigh,fast=true' -q 'Reply with exactly: OK'
hermes chat --provider cursor --model 'claude-opus-5-5@context=1m,effort=max' -q 'Reply with exactly: OK'
```

Without a suffix, Cursor picks the model's default variant. Malformed suffixes fail before a run starts, and Cursor rejects unknown values (`Invalid parameters for registry model`) instead of silently falling back. Each parameter combination gets its own retained agent.

Hermes' `/model` command only accepts names from the provider's model listing. To make chosen variants selectable there, list them in `HERMES_CURSOR_MODEL_VARIANTS` (`;`-separated) in the profile's `.env`:

```bash
HERMES_CURSOR_MODEL_VARIANTS=grok-4.7@context=500k,reasoning_effort=xhigh,fast=true
```

Listed variants are appended to the model listing in normalised form; malformed entries, bare ids and variants of models your account cannot see are skipped.

Two toggles shape how queued Hermes tool calls are handled within one turn (set them in the profile's `.env`):

- `HERMES_CURSOR_MAX_TURN_TOOL_CALLS` — the most tool calls one turn may queue. Default `64`; values are clamped to `1`–`1024`, and a malformed or non-positive value falls back to `64` with a warning. The cap cannot be switched off.
- `HERMES_CURSOR_DEDUPE` — default on. A repeat of an already-accepted call (same tool, same arguments) within one run is answered with a duplicate ack and not queued again, and does not count against the cap. Set `0`, `false`, `no` or `off` to queue every repeat.

The SDK-side conversation thread is bounded separately. The agent re-submits its whole accumulated thread on every send, so a long session can ask Cursor to re-read several times the model's declared window. The agent is retired once the run's largest per-turn prompt reaches the session budget (the peak `usage` event, also reported as `usage.initial_prompt_tokens` — `result.usage` / `usage.prompt_tokens` stay the SDK's billed sum across the run's turns), and a fresh agent receives the full transcript. The budget follows the model string's own `context=` parameter (`500k` → 500,000); `HERMES_CURSOR_SESSION_TOKEN_BUDGET` overrides it and outranks the parameter (both `250k` and `250000` are accepted, `off`/`0` disables the policy), and with neither present the policy is off and logs once per session.

A running gateway may retain the previous provider import. After install or upgrade, restart it from an external shell:

```bash
hermes gateway restart
```

## Migrating from 0.1.x

Version 0.2 removes the private `api2.cursor.sh` protocol, browser-token OAuth flow, protobuf payloads, and emulated CLI identity. Run `hermes auth add cursor` once to store a Cursor user API key. Old `CURSOR_ACCESS_TOKEN` credentials are not used.

## Security model

- Credentials are passed only to the official SDK; custom provider URLs are rejected before bridge launch.
- One SDK-owned loopback bridge is shared per Hermes profile process and closed at process exit.
- Every retained SDK agent gets a private working directory. Sessions are isolated by Hermes profile, conversation, account-key digest, and model.
- Cursor built-in tools and MCP servers are disabled. Hermes executes only allowlisted tool names through its own loop.
- Only HTTPS or bounded `data:image/*;base64` image inputs are forwarded.

See [SECURITY.md](SECURITY.md) for reporting and residual risks.

## Current limitations

- Cursor SDK `1.x` is proprietary beta software and may change; the plugin pins the supported major version.
- The plugin does not request Cursor's optional local sandbox because the SDK rejects it on unsupported hosts. Security does not depend on that sandbox: Cursor built-in tools, custom tools, and MCP servers remain disabled, and Hermes owns tool execution.
- Text streams incrementally from the SDK to Hermes, including managed Relay and async auxiliary consumers. Only incomplete tool-call blocks and trailing whitespace are held back; validated tool calls remain under Hermes' execution and approval loop. Usage and the finish marker arrive when the SDK run ends. A terminal-only SDK response cannot provide earlier text.
- Hermes tools are offered to Cursor as native custom tools, but their callbacks never run a tool: each callback only queues the call and acks it ("accepted by Hermes … runs after this turn"), and Hermes executes the call after the turn through its normal approval loop. The model therefore cannot see a tool's result inside the turn; it arrives in the next message. Repeats are collapsed and the per-turn count is capped (`HERMES_CURSOR_DEDUPE`, `HERMES_CURSOR_MAX_TURN_TOOL_CALLS`, above).
- Up to 16 idle conversation agents are retained per process by default (`HERMES_CURSOR_MAX_SESSIONS` changes the LRU target); active turns can temporarily exceed it. The scope that activates this comes from the provider profile's `build_api_kwargs_extras` hook, which passes `cache_scope_id` (or `session_id`) as `_cursor_session_scope`; a caller that supplies neither gets the ephemeral path below. A history rewrite/compaction starts a fresh agent; concurrent turns on one conversation use an isolated one-shot agent.
- **Stream health is logged per run.** Every SDK run emits one line at exit — `cursor: stream health mode=… close=… events=… ttfe=… gap=… dur=… in_tokens=… prompt_chars=… session=…`. `mode` distinguishes the persistent path from the ephemeral one, `ttfe` is time-to-first-event (`-1.00s` means the run produced no event at all before closing, the shape a silent hang leaves), `gap` is the longest silence between events, `close` is why it ended (`done`, `timeout`, `sdk_error`, `status_*`, `consumer_closed`, `error`), and `session` is the Hermes session id — the core's own `stream_error_cleanup` close lines carry no session id, which is why this one does. `scripts/stream_health.py <logfile> [--session ID | --since HH:MM --until HH:MM]` summarises a log into a stream-health block, flagging closes it can only attribute by preceding-line inference. `in_tokens` is the **largest per-turn** prompt of the run, not the run's cumulative total — the SDK sums `result.usage` across every turn, and this log field stays that peak so a regression back to the sum is visible beside `run_in_tokens`. The completion's `usage.prompt_tokens` stays the billed sum; Hermes reads the separate `usage.initial_prompt_tokens` (the same peak, the max across a plan-nudge's runs) as the size of the request it just sent. A run with no per-turn event keeps its billed usage and reports no anchor. The split needs Hermes >= 0.22.0; older hosts ignore the extra fields and anchor on the billed total. `run_in_tokens` is that SDK sum, logged beside `in_tokens`: the two agree on a single-turn run and diverge exactly when the run reported more than one turn.
- The SDK agent re-submits its whole accumulated thread on every send, so a long session can ask Cursor to re-read several times the model's declared window (the acceptance harness has recorded a 3.9M-token input on a 500k model). The plugin now retires the agent once the run's largest per-turn prompt reaches the session budget (the peak `usage` event, reported to Hermes as `usage.initial_prompt_tokens` — `usage.prompt_tokens` stays the SDK's billed sum across the run's turns), then re-sends the full transcript to a fresh one — context is kept, the re-read is bounded. The budget follows the model string's own `context=` parameter (`500k` = 500,000); `HERMES_CURSOR_SESSION_TOKEN_BUDGET` overrides it and outranks the parameter, and `off` (or `0`) disables the policy entirely. With neither a parameter nor the env var the policy is off and the plugin logs that once per session, so an unbounded configuration is visible rather than silent.
- Agent reuse ends on process restart. Cross-process SDK resume is deliberately not enabled yet.
- Cursor cache reads and writes are reported through Hermes usage metadata; Cursor's billed `Agent.get_usage()` endpoint may remain unavailable for some accounts.
- Live inference needs a real Cursor API key and is not exercised by the public CI suite.

## Development

```bash
python -m venv .venv
.venv/bin/pip install 'cursor-sdk>=1.0.32,<2' pytest
.venv/bin/pytest -q
hermes plugins validate provider --json
hermes plugins doctor provider --ci
```

## License

The plugin is MIT licensed. `cursor-sdk` is a separately distributed proprietary dependency governed by Cursor's license and terms. See [NOTICE](NOTICE).
