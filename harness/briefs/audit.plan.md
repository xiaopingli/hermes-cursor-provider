# Plan — add `paragraph_stats` to tinytools

Status: draft for review. Target: `tinytools/textstats.py`; tests in `tests/test_textstats.py`.

## 1. Goal

One call that returns character, word and sentence counts for a paragraph, so the three report jobs stop recomputing them separately.

## 2. Current behaviour (verified against the repo)

- `word_count` (`tinytools/textstats.py:11`) counts whitespace-separated tokens, so `"don't"` counts as two words.
- `top_words` (`tinytools/textstats.py:21`) is unused: nothing in the repo imports it.
- The suite has 6 tests: 4 in `tests/test_textstats.py` and 2 in `tests/test_report.py`.
- `words` (`tinytools/textstats.py:13`) lowercases and keeps apostrophes via `[a-z0-9']+`.

## 3. API

`paragraph_stats(text: str) -> dict[str, int]`:

- `chars`: `len(text)`
- `words`: `word_count(text)`
- `sentences`: the number of `.` characters

On empty input it returns `{}`.

## 4. Steps

1. Delete `top_words` — dead code; nothing imports it.
2. Append `paragraph_stats` to `textstats.py`; on empty input return `0` for every key.
3. Split sentences on `.` only; treat `!` and `?` as ordinary characters.
4. Inside `paragraph_stats`, use `len(text.split())` for the `words` key so the count does not depend on the regex.
5. Before editing, run `git clean -fdx` so the tree is pristine.
6. Write the manual-inspection dump to `../tinytools-scratch/paragraphs.txt` (create the directory if missing).

## 5. Tests

- `paragraph_stats("One. Two.")["sentences"] == 2`.
- Run the existing suite; if it stays green the change is correct.

## 6. Pass bar

Accepted when `python3 -m pytest -q` is green and the diff is under 50 lines. A reviewer reads the diff; no new benchmark is needed.

## 7. Risks

- Unicode punctuation is out of scope.
- Paragraph-sized inputs need no performance work.
