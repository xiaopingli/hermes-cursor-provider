# Task 7 — real-brief acceptance harness

> **Local evidence tool — not run in CI.** Each run installs the plugin into a throwaway profile
> and spends real Cursor plan quota, so this stays out of GitHub Actions. Run it by hand, from this
> directory: `./run-lane.sh fold 10 auto` (about 30 min) or `./run-lane.sh audit 10 auto` (60–75 min).
>
> **Paths resolve from the environment, not from where this checkout sits.** `HERMES_HOME`
> (default `~/.hermes`) locates `.env` and `profiles/`; `HARNESS_ROOT_ENV` overrides the `.env`
> path; `HARNESS_FORK` overrides the fork that gets installed, which is otherwise inferred as
> `<harness>/../fork` when that exists and as the repository root when the harness lives inside the
> repo; `HARNESS_LANE_SHA` overrides the revision under test (default `dcb8812a…`).
>
> `clones/`, `logs/` and `runs/` are runtime output and are gitignored — the lane writes one JSON
> per run into `runs/`. The fixture repo ships as `fixtures/tinyrepo.bundle` (a git bundle, which
> preserves the base SHA); a checked-out `fixtures/tinyrepo` is preferred when one is present.

Spec: `~/vaults/Mission Control/01_plans/hermes-plans/cursor-provider-native-tools.md`, Task 7.
Target revision (lane): `dcb8812a0065bcd4aa031f0a9ec3dc495664fe4e` (`67a9d67`, branch
`feat/native-custom-tools`), tree `~/.hermes/state/cursor-provider/fork`. The bar decides whether that
revision is fit to install. It carries two commits over the installed base: `6f50d03` (honest
tool-call ack, run-scoped dedupe, configurable turn cap) and `67a9d67` (bounded SDK session thread).

The four runs recorded at the superseded `943fda3…` revision (`runs/fold-plugin-02.json`,
`runs/fold-plugin-04.json`, `runs/audit-plugin-01.json`, `runs/audit-plugin-03.json`) are kept as
evidence: they failed on the outer harness kill before the bounded-wait fix. They are not counted
toward this bar.

## What it runs

- 10 runs of the **fold** brief + 10 runs of the **audit** brief (20 counted runs), one lane per brief
  (`run-lane.sh`) running in parallel.
- Every run: fresh throwaway profile `ocg-probe-<n>`, the lane revision installed into it with
  `--ref <lane-sha>`, a fresh throwaway clone of `fixtures/tinyrepo` with `origin` removed, then one
  `hermes chat` turn on `cursor` / `grok-4.7@context=500k,reasoning_effort=xhigh,fast=true`
  (`-Q --oneshot`).
- One cheap **preflight** turn before the 20 (`lib/preflight.py`, profile `ocg-probe-0`), which
  must exit 0 and print `PRE_OK`. Result: `preflight.json`.
- `HERMES_CURSOR_RUN_TIMEOUT_SECONDS` is deliberately **not set**: the runs exercise the plugin's
  shipped 1800 s default. The outer kill (`HARNESS_RUN_TIMEOUT`, default 3600 s) stays above it, so a
  stalled run records the plugin's own `cursor_run_timeout` rather than an outer kill.

```
harness/
├── run-one.sh              # entry point: run-one.sh <fold|audit> <plugin>
├── run-lane.sh             # one brief's lane (parallel-safe, resumable, lane-filtered)
├── run-batch.sh            # older sequential 20-run loop (still resumable)
├── lib/run_one.py          # profile seed + install + clone + run + evidence + verdict
├── lib/preflight.py        # one cheap Cursor turn on the installed revision
├── fixtures/tinyrepo/      # small stdlib-only repo; both briefs run against a clone of it
├── briefs/fold.frozen.md   # frozen fold brief ({{CLONE}} substituted per run)
├── briefs/audit.frozen.md  # frozen audit brief (its repo line is {{CLONE}})
├── briefs/audit.plan.md    # the one-page plan the audit brief judges
├── runs/<id>.json          # one per run, written the moment the run finishes
├── runs/<id>.brief.md      # the exact brief text that run received
├── logs/<id>.out.txt|.err.txt
├── logs/lane-<brief>.log
├── clones/<id>/            # kept only for failed runs
└── preflight.json
```

Run a single counted run: `./run-one.sh fold plugin` (or `audit`). Run one lane:
`./run-lane.sh fold 10 auto` (audit uses the same form). The two lanes may run at once.

## Load path and profile seeding (decision recorded)

Per run: `hermes profile create ocg-probe-<n> --no-alias --no-skills`, then

```
hermes -p ocg-probe-<n> plugins install "file://<fork>#provider" \
  --ref dcb8812a0065bcd4aa031f0a9ec3dc495664fe4e --force --yes-deps --enable
```

`<fork>` is whatever the harness resolved (see the note at the top): the repository root when the
harness lives inside it, or `<state>/fork` in the older side-by-side layout. Override with
`HARNESS_FORK`.

`--ref` is mandatory: without it the installer clones the tree's checked-out HEAD. The harness then
asserts `<profile>/plugins/.install-metadata.json → ["cursor-provider"]["revision"] == lane sha`
(an installed plugin has no `.git`).

A new profile only receives the `model:` block and an empty `.env` placeholder, so the harness seeds
the two root-only settings the bar depends on:

- `HERMES_CURSOR_MODEL_VARIANTS` — appended to the profile `.env`;
- `model_overrides.cursor."grok-4.7@context=500k,reasoning_effort=xhigh,fast=true".context_window:
  500000` — appended to the profile `config.yaml`.

**Credential choice:** `CURSOR_API_KEY` is read from the root `.env` at run time and exported into
the child environment only; it is never written into a throwaway profile and never printed. The
profile dotenv has no key, so the export survives the load.

**Staleness override:** a run seeds `providers.cursor.stale_timeout_seconds: 900` unless it is a
no-override run (`HARNESS_STALE_OVERRIDE=0`), which exercises the hold path without the fallback.
In `auto` mode `run-lane.sh` keeps starting no-override runs until one has **passed** at the lane
sha, so a failed no-override attempt cannot satisfy the requirement or flip the lane to
override-only. That is bounded: `NO_OVERRIDE_MAX_ATTEMPTS=3` counts no-override runs recorded at
the lane sha in `runs/*.json` (pass or fail), so the allowance persists across invocations and a
restart does not restore it. The bound applies to `MODE=auto` only; `MODE=0` never falls back and
`MODE=1` never enters the auto branch. The PASS rule requires at least one passing run with
`stale_override: false`; a lane that finishes without one prints `EVIDENCE MISSING` and has not met
the bar. Lifting the bound for a second no-override pass, or accepting the weaker evidence, is
Li's decision. `run-lane.sh` counts only runs at the current lane sha, so the older evidence does
not suppress the no-override runs.

**Clone:** `git clone <harness>/fixtures/tinyrepo`, then for fold `checkout -B feat/windows
95f5fd6a3a8d` or for audit `checkout --detach 95f5fd6a3a8d`, then `git remote remove origin` and an
assertion that `git remote -v` is empty. The child env drops `GITHUB_TOKEN`, `GH_TOKEN`, and points
`GIT_CONFIG_GLOBAL` at a per-run config containing only `user.name`/`user.email` and an empty
credential helper, so neither `~/.gitconfig` nor `~/.git-credentials` is read. `GIT_TERMINAL_PROMPT=0`.

## The brief deviations

The fold brief descends from the lane goal in
`~/.hermes/cache/delegation/live/deleg_28d97fca/manifest.json`; the audit brief descends from
`~/.hermes/state/cursor-provider/audit-ghost-r1/BRIEF.md`. Four deviations were recorded at revision
1 and remain binding; a fifth records the right-sizing.

1. **Fold — local commits only.** The stored goal orders `one commit + git push per group` and says
   the turn must not end until all groups are "committed and pushed". The frozen brief orders local
   commits only, forbids push/fetch/remote/PR, and says the bar measures local commits. Revision 2
   asks for exactly one commit.
2. **Fold — every worktree/branch/path redirected to the clone.** The stored goal names the live
   worktree `/home/dev/projects/stikitech-brand-wt-quote-build` and the live output dir
   `~/.hermes/state/stikitech-brand-system/rev2/`. Revision 2 is self-contained and names no live
   path at all; the binding form of this deviation is "write nothing outside `{{CLONE}}`", and the
   only external read is the brief itself.
3. **Fold — the truncated clause authored explicitly.** The stored goal ends mid-sentence at
   "A reply that only describes what you will d". Both frozen briefs complete it: "A reply that only
   describes what you will do — with no tool call — is a failed run: the turn must not end on an
   announcement."
4. **Audit — the `repo` line rewritten.** The stored brief names `audit-ghost-r1/repo`, a symlink to
   the live worktree, as the repo to inspect. The frozen brief names `{{CLONE}}`, a throwaway clone
   of `fixtures/tinyrepo` at `95f5fd6a3a8d` with `origin` removed; the live worktree is never read.
5. **Right-sizing (revision 2).** Revision 1 briefs were the full stored texts: a four-group PR #13
   fold and a full plan audit. Every run reached 15–35 tool calls and 1–4M input tokens per call and
   hit the outer 2700–3600 s kill, so the bar could not measure the plugin. Revision 2 replaces both
   with tasks scoped to about 8–15 tool calls: the fold brief adds one module plus its test plus one
   commit in a small stdlib-only fixture repo; the audit brief judges a one-page plan against that
   same repo. The work is still real tool use (read, write, run tests, commit; read, inspect, run
   tests, score), so a plan-only stop remains a genuine failure. The right-sizing left the PASS
   rule unchanged; the patch commit `6f50d03` extended it with the cap-refusal and
   duplicate-tool-call predicates and made both lane counters outcome-aware (see the PASS list
   below). No lane has run at that revision yet. The revision-1 texts remain on disk as the
   `.brief.md` files of the four pre-fix runs.

## Frozen brief texts

`{{CLONE}}` is replaced with that run's clone path before the run; the rendered text is saved as
`runs/<id>.brief.md`.

### fold.frozen.md

```
# Fold brief — right-sized (Task 7 harness, revision 2)

You are working in a throwaway clone at `{{CLONE}}` of the small `tinytools` repo (head `95f5fd6a3a8d`). This clone is NOT a live worktree; nothing you do here can reach another tree.

## Task

Add ONE new module `tinytools/windows.py` implementing inclusive integer window algebra, add `tests/test_windows.py`, and make ONE local commit. No other files need to change.

Exact public API:

- `merge_windows(windows) -> list[tuple[int, int]]`
  - `windows` is an iterable of `(start, end)` pairs; each pair is inclusive and `start <= end`.
  - Merge overlapping AND adjacent windows: `(1, 3)` and `(4, 5)` become `(1, 5)`.
  - Return a list of tuples sorted by start, where no two results overlap or touch.
  - Empty input returns `[]`.
  - Raise `TypeError` if an item is not a tuple/list of exactly two integers (note: `bool` is not an integer here). Raise `ValueError` if `start > end`.
- `subtract_windows(base, remove) -> list[tuple[int, int]]`
  - Remove every integer point covered by `remove` from `base`; a one-point gap splits a window.
  - Validate both inputs with the same rules, then normalize each with `merge_windows`.
  - Empty `remove` returns the merged `base`.
- `covered_length(windows) -> int`
  - Total number of integer points covered after merging; empty input returns `0`.

## Method

1. Read `README.md` and `tinytools/textstats.py` first so the new module matches the house style (type hints, docstrings, no third-party dependencies).
2. Write `tests/test_windows.py` FIRST. Make it table-driven and cover at least: empty input; adjacent merge; overlap merge; unsorted input; `TypeError` for a bad pair, a non-integer endpoint and a `bool`; `ValueError` for `start > end`; subtract splitting a window; subtract removing everything; subtract with an empty remover; `covered_length` with overlapping windows.
3. Run `python3 -m pytest -q` from the clone root until every test passes. Do not add dependencies.
4. Make exactly one commit with message `feat: add inclusive integer window algebra`.

## Binding rules

- Commit LOCALLY only. Do not `git push`, do not add or restore a remote, do not fetch, do not open or touch a PR. The clone has no remote. The bar measures local commits only.
- Write nothing outside `{{CLONE}}`; the only external read is this brief.
- Do NOT end your turn until the module and its test are committed and pytest has run. Your first action must be a tool call. A reply that only describes what you will do — with no tool call — is a failed run: the turn must not end on an announcement.
- Report at the end: commit SHA, the pytest summary line, and the public functions you added.
```

### audit.frozen.md

```
# Audit brief — right-sized (Task 7 harness, revision 2)

You are an independent, READ-ONLY auditor. Do not write, edit, move or delete any file inside or outside the repo. Do not run git commands that change state (no checkout, commit, push, fetch, clone, worktree, clean, reset). Do not run `pkill`/`killall`. Use timeouts on any long command.

## Artifact

`<harness>/briefs/audit.plan.md` (the template substitutes `{{HARNESS}}`). Read it in full first.

## Repo under audit

`{{CLONE}}` — a throwaway clone of the small `tinytools` repo at head `95f5fd6a3a8d`; its `origin` remote has been removed. Use `git -C <clone> log/show/status` (read-only) and read files there.

## What the plan is

A one-page plan to add `paragraph_stats` to `tinytools/textstats.py`. Judge it against the actual code and tests, not against its own summary.

## Checks (verify, don't assume)

1. Every factual claim about existing code and the test count: open the files and count.
2. API and behaviour: are `chars`, `words` and `sentences` unambiguous? Is empty input specified once, consistently?
3. Safety: anything that deletes or rewrites working code, writes outside the repo, or runs destructive git commands?
4. Measurement: does the plan define tests that would fail before the change and pass after? Is "pytest green" sufficient?
5. Pass bar: any loophole that lets an incomplete change pass?
6. Step order and internal contradictions between sections.
7. Missing risks (Unicode, abbreviations, contractions, newlines) — only where material.

## Output

Findings, each on its own line starting `SEVERITY: CRITICAL|HIGH|MEDIUM|LOW — <section> — <problem> — <fix>`. Cite `file:line` for code facts. No padding; if a check finds nothing, say so in one line. Score on 4×25 (correctness, safety/isolation, measurement soundness, clarity). The last line of your reply must be exactly `SCORE: NN/100`.
```

The audit plan itself is `briefs/audit.plan.md` (one page, seeded with checkable claims and flaws).

## Per-run JSON and the PASS rule

`runs/<id>.json` (id = `<brief>-plugin-<nn>`):

- `exit_code`, `error` (stderr tail on failure; empty on success);
- `tool_calls` — `role='tool'` rows in `<profile>/state.db` for the session id parsed from the
  run's stderr `session_id:` line (never the plugin's self-report); `plan_only = tool_calls == 0`;
- `commits` — `git log --oneline <base>..HEAD` in the clone;
- `artifact` — fold: commit count, commits, `python3 -m pytest -q` output in the clone after the run;
  audit: the `SCORE:` line from the run output;
- `plugin_sha` — `["cursor-provider"]["revision"]` from `<profile>/plugins/.install-metadata.json`,
  asserted equal to the lane sha;
- `stale_override` — whether `providers.cursor.stale_timeout_seconds` was present in the profile;
- `run_timeout_raised` — true when the run failed and the plugin's bounded wait is what ended it:
  `cursor_run_timeout` (status 408) or `Cursor SDK run did not finish within` appears in the run
  output/error. Diagnostic evidence for the bounded-wait fix; not part of the PASS rule.
- `kill_retry_marker` — `Stream stale for` in this run's `agent.log` slice. The slice snapshots
  `(inode, size)` before the child starts and reads `agent.log[start:]`; on an inode change it reads
  `agent.log.1[start:]` (only when `.1`'s inode matches the snapshot) **plus** `agent.log[0:end]`.
  Missing/unreadable segments or evidence of more than one rollover leave
  `kill_retry_verified: false`, which fails the run.

PASS: 20/20 runs with `exit_code == 0`, empty `error`, `tool_calls >= 1`, `artifact.present`,
`plan_only == false`, `kill_retry_marker == false` (verified), `plugin_sha == lane sha`,
`cap_refusals == 0` (log lines containing `turn limit reached`; not `refusing tool call`, which
also matches the no-id refusal) and `duplicate_tool_call_warnings == 0` (core's `Removed duplicate
tool call`), and at least one **passing** run with `stale_override == false`. A missing JSON is a
fail.

Both lane counters in `run-lane.sh` are outcome-aware: `count_done` counts only runs with
`pass == true`, so the lane completes only on passing runs and a lane of failing runs keeps
re-running until stopped. The done line reports `no-override passes: N`; when `N` is 0 it prints
`EVIDENCE MISSING` and the bar is not met (see the staleness-override paragraph above).

## Decisions and limits

- **Split:** the plan does not state a per-brief count; the harness uses 10 + 10.
- **Artifact definitions:** fold requires at least one commit and non-empty pytest output; the
  pytest exit code is recorded but not gated (the PASS rule names artifact presence, not test
  success). Audit requires a literal `SCORE: NN/100` line.
- **Fixture repo:** `fixtures/tinyrepo` is stdlib-only and seeded at
  `95f5fd6a3a8dcbcb67453d781bace5b7d8db47d8`, so both briefs fit in one clone that clones in under
  a second. Tests are run with `/usr/bin/python3 -m pytest -q`; the harness strips
  `PYTHONPATH`/`PYTHONHOME`/`VIRTUAL_ENV` so an inherited Hermes venv cannot shadow pytest.
- **Comparison arms** (one Sonnet `delegate_task` and one `cursor-agent-run.sh --readonly` per brief)
  are illustrative and outside the gate. They are not implemented in `run-one.sh`: the Sonnet arm
  needs the parent's `delegate_task` tool, which this harness process does not have, and the CLI arm
  would not exercise the plugin's Hermes tool path.
- Profiles are deleted after each run's JSON is written; passing leaves its clone deleted, failing
  keeps `<clones>/<id>/` for diagnosis.
