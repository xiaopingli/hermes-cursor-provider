# Fold brief — right-sized (Task 7 harness, revision 2)

You are working in a throwaway clone at `{{CLONE}}` of the small `tinytools` repo (head `95f5fd6a3a8d`). This clone is NOT a live worktree; nothing you do here can reach another tree.

## Environment facts (read first)

- The `terminal` tool starts in the home directory (`{{HOME}}`), NOT in the clone. Prefix every shell command with `cd {{CLONE}} && `.
- The `python3` on PATH is a Hermes runtime without pytest. Use `/usr/bin/python3 -m pytest -q` for tests.
- `execute_code` is unavailable in this mode. Use `read_file`, `write_file`, `search_files` and `terminal`.
- Never repeat a tool call with identical arguments. If a command fails, change it before retrying.

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
3. Run `cd {{CLONE}} && /usr/bin/python3 -m pytest -q` until every test passes. Do not add dependencies.
4. Make exactly one commit with message `feat: add inclusive integer window algebra`.

## Binding rules

- Commit LOCALLY only. Do not `git push`, do not add or restore a remote, do not fetch, do not open or touch a PR. The clone has no remote. The bar measures local commits only.
- Write nothing outside `{{CLONE}}`; the only external read is this brief.
- Do NOT end your turn until the module and its test are committed and pytest has run. Your first action must be a tool call. A reply that only describes what you will do — with no tool call — is a failed run: the turn must not end on an announcement.
- Report at the end: commit SHA, the pytest summary line, and the public functions you added.
