# Audit brief — right-sized (Task 7 harness, revision 2)

You are an independent, READ-ONLY auditor. Do not write, edit, move or delete any file inside or outside the repo. Do not run git commands that change state (no checkout, commit, push, fetch, clone, worktree, clean, reset). Do not run `pkill`/`killall`. Use timeouts on any long command.

## Environment facts (read first)

- The `terminal` tool starts in the home directory (`{{HOME}}`), NOT in the clone. Use absolute paths, e.g. `git -C {{CLONE}} log`, and `cd {{CLONE}} &&` before any repo-relative command.
- `execute_code` is unavailable in this mode. Use `read_file`, `search_files` and `terminal`.
- Never repeat a tool call with identical arguments. If a command fails, change it before retrying.

## Artifact

`{{HARNESS}}/briefs/audit.plan.md`. Read it in full first.

## Repo under audit

`{{CLONE}}` — a throwaway clone of the small `tinytools` repo at head `95f5fd6a3a8d`; its `origin` remote has been removed. Use `git -C {{CLONE}} log/show/status` (read-only) and read files there.

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
