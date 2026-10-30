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

Cursor's SDK is an agent surface, not a raw chat-completions API. The plugin retains one SDK agent per Hermes conversation/account/model, sends only newly appended messages, and translates explicit `<tool_call>` blocks back into Hermes tool calls. SDK custom tools stay off unless `HERMES_CURSOR_NATIVE_TOOLS` is set or the model name says `native_tools=on`; see [Native tools (opt-in)](#native-tools-opt-in).

## Native tools (opt-in)

Offer Hermes's tool list as Cursor custom tools on the stream path in either of two ways. The model name wins over the environment in both directions.

- Model parameter: append `native_tools=on` or `native_tools=off` to the `id@key=value,…` suffix, for example `grok-4.7@context=500k,reasoning_effort=xhigh,fast=true,native_tools=on` or `grok-4.7@native_tools=on`. `native_tools` is plugin-only and is not sent to Cursor. A model whose only parameter is `native_tools` selects the bare id. Any other value is an error. Switching it with `/model` starts a different retained session.
- Environment: `HERMES_CURSOR_NATIVE_TOOLS=1` turns it on when the model name does not say. `=0` or unset leaves it off unless the name says `native_tools=on`.

They ride the SDK `mcp` capability group (`tools: ["mcp"]`). No MCP servers are loaded: `mcp_servers` stays empty and `setting_sources` is not set, so the only tools are the ones Hermes offered. The executor is inert. Hermes still decides and runs every tool after the turn; the model must not invent a result.

`HERMES_CURSOR_NATIVE_MAX_CALLS` caps accepted calls in one run (default 64, clamped 1–1024). A non-stream request (streaming off, or Hermes's one-off 5xx retry) stays on the upstream text protocol even when the name says `native_tools=on`. The first such request in a native session re-sends the full transcript, and the next stream request re-sends it again. In a measured comparison the native arm took about twice the wall time and more tool rounds than the text protocol. A native run is cancelled when Cursor starts a second answer; the session agent is retired and the next turn re-sends the full transcript, and usage is the sum of turns up to the cut, so Cursor's dashboard can show slightly more than Hermes reports.

This host runs one gateway that serves every profile (`gateway.multiplex_profiles`, default on). Under that gateway:

1. A named profile's own `.env` does not reach the plugin, so use the model parameter there.
2. The env form in the launch (default) `.env` turns native mode on for every profile that gateway serves.
3. To pick a `…,native_tools=on` name with `/model`, list it in the launch `.env`'s `HERMES_CURSOR_MODEL_VARIANTS`. CLI `--model` and a config model string are not catalog-checked.

Rollback: remove `native_tools=on|off` from every model name (and from `HERMES_CURSOR_MODEL_VARIANTS`) first. Unsetting `HERMES_CURSOR_NATIVE_TOOLS` only covers the env form.

## Install

```bash
hermes plugins install cursor-provider
hermes auth add cursor
hermes model
```

Create a **user API key** at <https://cursor.com/dashboard/api>. User and service-account API keys are supported by Cursor's SDK; Team Admin keys are not inference credentials.

Plugins and credentials are profile-scoped:

```bash
hermes --profile work plugins install cursor-provider
hermes --profile work auth add cursor
```

Test without changing the configured default model:

```bash
hermes chat --provider cursor --model auto -q 'Reply with exactly: OK'
```

### Model parameters (effort, fast, context)

Cursor models expose parameters such as reasoning effort, fast mode and context size (see each model's `parameters` in the SDK's `list_models()`). Append them to the model name as `@key=value,...`:

```bash
hermes chat --provider cursor --model 'grok-4.7@fast=true,reasoning_effort=xhigh' -q 'Reply with exactly: OK'
hermes chat --provider cursor --model 'claude-opus-5-5@context=1m,effort=max' -q 'Reply with exactly: OK'
```

Without a suffix, Cursor picks the model's default variant. Parameter ids and values are checked against your account's `list_models()` catalog before an agent is created, so a malformed or unsupported selection fails with an error instead of silently using the default variant. The catalog must be reachable for a parameterised run. Each parameter combination gets its own retained agent; parameter order does not matter. A `context=` parameter also sets the context budget Hermes uses for compression.

`HERMES_CURSOR_SESSION_TOKEN_BUDGET` bounds the SDK session thread. The agent re-submits its whole accumulated thread on every send, so a long session can ask Cursor to re-read several times the model's window. The default budget is that model's own `context=` parameter (`500k` = 500,000); the env value outranks the parameter. `off` or `0` disables the policy. When the last call's reported input reaches the budget the retained agent is retired and the next turn re-sends the full transcript; the fresh agent logs `reason=budget`. With neither the env var nor a `context=` parameter the policy is off and the plugin logs that once per session.

Hermes' `/reasoning` and `delegation.reasoning_effort` settings are **not** mapped onto Cursor parameters automatically. Select the model's own parameter explicitly with the `@key=value` suffix (for example `grok-4.7@reasoning_effort=xhigh`).

Hermes' `/model` command only accepts names from the provider's model listing. To make chosen variants selectable there, list them in `HERMES_CURSOR_MODEL_VARIANTS` (`;`-separated) in the launch (default) profile's `.env`. Under a multiplexing gateway a named profile's own `.env` does not reach the plugin:

```bash
HERMES_CURSOR_MODEL_VARIANTS=grok-4.7@context=500k,reasoning_effort=xhigh,fast=true
```

Listed variants are appended to the model listing in normalised form; malformed entries, bare ids, variants of models your account cannot see, and unsupported parameter ids or values are skipped. If the picker already cached the model list, run `hermes model --refresh` after changing `HERMES_CURSOR_MODEL_VARIANTS`.

A running gateway may retain the previous provider import. After install or upgrade, restart it from an external shell:

```bash
hermes gateway restart
```

## Migrating from 0.1.x

Version 0.2 removes the private `api2.cursor.sh` protocol, browser-token OAuth flow, protobuf payloads, and emulated CLI identity. Run `hermes auth add cursor` once to store a Cursor user API key. Old `CURSOR_ACCESS_TOKEN` credentials are not used.

## Troubleshooting

If `hermes model` lists only `auto` and `composer-2.5`, the account model list did not load. Hermes logs a warning with the reason. The most common cause is that `cursor-sdk` is not installed in the Python environment Hermes runs with. Install it there (`cursor-sdk>=1.0.32,<2`), then run `hermes model --refresh`, because the picker caches the seed list.

## Security model

- Credentials are passed only to the official SDK; custom provider URLs are rejected before bridge launch.
- One SDK-owned loopback bridge is shared per Hermes profile process and closed at process exit.
- Run-scoped `CancelRun`/`WaitLiveRun` use the SDK's local-bridge opt-in because the SDK guards them as cloud calls and its client exposes no per-call key for them. The scoped account key is still sent explicitly on `CreateAgent`, `ListModels`, and `Send`, and an empty scoped key fails before bridge launch, so a process `CURSOR_API_KEY` is never used for a user-scoped turn.
- Every retained SDK agent gets a private working directory. Sessions are isolated by Hermes profile, conversation, account-key digest, and model.
- Cursor built-in tools stay disabled, and no MCP servers are loaded. Custom tools stay off unless native tools are enabled; even then the executor is inert and Hermes runs every call. Hermes executes only allowlisted tool names through its own loop.
- Only HTTPS or bounded `data:image/*;base64` image inputs are forwarded.

See [SECURITY.md](SECURITY.md) for reporting and residual risks.

## Current limitations

- Cursor SDK `1.x` is proprietary beta software and may change; the plugin pins the supported major version.
- Model parameters are explicit only: Hermes' `/reasoning` and `delegation.reasoning_effort` settings are not mapped automatically onto Cursor's `reasoning_effort`/`effort` parameters.
- The plugin does not request Cursor's optional local sandbox because the SDK rejects it on unsupported hosts. Security does not depend on that sandbox: Cursor built-in tools stay disabled and no MCP servers are loaded. Custom tools stay off unless native tools are enabled; even then the executor is inert and Hermes runs every call.
- Text streams incrementally from the SDK to Hermes, including managed Relay and async auxiliary consumers. Incomplete tool-call blocks, trailing whitespace, and a possible short action-only announcement are held back. Required/named tool-choice responses and corrective attempts are held until a validated call arrives; ordinary text still streams incrementally. Usage and the finish marker arrive when the SDK run ends. A terminal-only SDK response cannot provide earlier text.
- A short, unconditional action announcement with no validated call triggers at most two corrective nudges, as does an unsatisfied required/named tool choice when tools are available. Failed preliminary text is not returned as a completed answer or saved in Hermes session history. Exhaustion raises `CursorSDKRunError`; the adapter never infers or executes a tool. Questions, conditional offers, ordinary answers, and requests without enabled tools are not retried. See `docs/recovery-tdd.md` for offline verification and scope.
- Tool calls use a prompt-level contract unless native tools are enabled, because the official SDK's custom-tool callbacks execute inside the SDK run rather than Hermes' normal approval loop and are not used as an executor. Custom tools stay off unless native tools are enabled; even then the executor is inert and Hermes runs every call. Offered Hermes calls emitted on the native channel are intercepted and returned to Hermes; Cursor-native execution remains disabled.
- Up to 16 idle conversation agents are retained per process by default (`HERMES_CURSOR_MAX_SESSIONS` changes the LRU target); active turns can temporarily exceed it. A history rewrite/compaction starts a fresh agent; concurrent turns on one conversation use an isolated one-shot agent.
- Agent reuse ends on process restart. Cross-process SDK resume is deliberately not enabled yet.
- Cursor cache reads and writes are reported through Hermes usage metadata; Cursor's billed `Agent.get_usage()` endpoint may remain unavailable for some accounts.
- `usage.prompt_tokens`, `completion_tokens`, and the cache details stay the SDK's **billed totals** (Cursor sums them across every turn, and this adapter adds them across recovery attempts). The separate context size is reported through `usage.initial_prompt_tokens` / `usage.initial_cached_tokens`, which Hermes treats as the size of the request it just sent; the value is the run's **largest single turn**, because a run's turns are not ordered by size (one live thread measured 14,191 then 8,686 on consecutive calls) and understating the context lets a session overflow before compression fires. Summing attempts into the anchor instead reported 723k for three 240k/241k/242k-token prompts and failed a session that fit its window closed as "too large to compress further". A run that emits no `usage` event keeps its `result.usage` and reports no anchor. The split needs Hermes >= 0.22.0; older hosts ignore the extra fields and anchor on the billed prompt total.
- One Cursor SDK run is bounded by `HERMES_CURSOR_RUN_TIMEOUT_SECONDS` (default `1800`, clamped `0.05`–`86400`; malformed, non-finite, or non-positive values fall back to the default, and the cap cannot be switched off). The deadline covers the blocking Send stream (including the SDK's priming read), every lazy stream pull, and the result wait — not just `run.wait()`. Each pull is bounded separately, so text still streams on demand and cancellation still follows the pull. A result that lands as the caller gives up is handed to exactly one owner, so a late `send()` still cancels its run. At the deadline a retryable `CursorSDKRunError` (status 408, `error_code=cursor_run_timeout`) is raised; Hermes classifies 408 as a timeout and retries it, but the provider makes no fallback promise. Cancel/close are best-effort with a short, cap-scaled grace per call, so cleanup can add that grace but not a full transport timeout. The wall-clock deadline includes consumer processing between stream pulls; existing SDK idle/unary timeouts can fail earlier. Remaining ceiling: a transport read already in flight cannot be interrupted; its worker is abandoned as a daemon thread and is reclaimed only when that call returns or times out (stream reads use the Hermes request timeout, 300 s by default; unary cleanup uses the SDK's 30 s bound), or when the server honours cancellation.
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
