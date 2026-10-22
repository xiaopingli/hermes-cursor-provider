#!/usr/bin/env python3
"""Mutation check: prove the test suite actually guards the behaviours we rely on.

Why this exists
---------------
Two real defects in this project's history were found by mutation testing and *not* by the
suite as written: a missing write-back of reported tokens, and a guard that computed a retire
decision and then ignored it. Both left every test green. A suite can only fail on a regression
it is actually sensitive to, and reading the tests does not tell you whether it is.

Each entry below breaks exactly one guarantee. The check applies it, runs the suite, and
requires the suite to go red. A mutation that leaves the suite green is a hole: that regression
would reach production unopposed.

Fails closed
------------
The source is backed up byte-for-byte before anything is touched, restored in a ``finally``, and
the restore is verified by sha256. A crash mid-run cannot leave a mutated file behind. Anchors
that do not occur exactly once are reported as pattern errors rather than silently skipped,
because an anchor that no longer matches means the mutation is no longer testing what it claims.

Usage
-----
    python scripts/mutation_check.py [--repo-root DIR] [-k NAME]

Exits 0 only when every mutation is caught and every anchor matched.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import shutil
import subprocess
import sys

CLIENT = "provider/hermes_cursor_provider/client.py"
TOOL = "scripts/stream_health.py"

# (name, file, old, new, why)
MUTATIONS: list[tuple[str, str, str, str, str]] = [
    # --- P1: the session token budget -------------------------------------------------
    (
        "unknown_retires",
        CLIENT,
        "    if budget is None or last_input_tokens <= 0:\n        return False\n"
        "    return last_input_tokens >= budget",
        "    if budget is None:\n        return False\n    if last_input_tokens <= 0:\n"
        "        return True\n    return last_input_tokens >= budget",
        "zero reported tokens means unknown; retiring on unknown throws away every agent "
        "whose usage the provider did not report",
    ),
    (
        "never_retire",
        CLIENT,
        "    return last_input_tokens >= budget",
        "    return False",
        "the bound must actually fire or it is decoration",
    ),
    (
        "always_retire",
        CLIENT,
        "    if budget is None or last_input_tokens <= 0:\n        return False",
        "    if False:\n        return False",
        "retiring every turn defeats the purpose (cheaper, but with no session reuse at all)",
    ),
    (
        "env_loses_to_model",
        CLIENT,
        '    raw = os.environ.get(_ENV_SESSION_TOKEN_BUDGET, "").strip()\n    if raw:',
        '    raw = ""\n    if raw:',
        "the operator's env override must outrank the model's declared window",
    ),
    (
        "k_is_1024",
        CLIENT,
        '        if text.endswith("k"):\n            return int(float(text[:-1]) * 1000)',
        '        if text.endswith("k"):\n            return int(float(text[:-1]) * 1024)',
        "500k must mean 500,000: a kibi reading silently shrinks the budget by 2.4%",
    ),
    (
        "no_write_back",
        CLIENT,
        "            session.last_input_tokens = _context_size_tokens(result, context_usage)\n",
        "",
        "the decision is only useful if the reported tokens are recorded for the next turn",
    ),
    (
        "guard_ignores_budget",
        CLIENT,
        "(not extends or toolset_changed or over_budget)",
        "(not extends or toolset_changed)",
        "a computed retire decision that the guard ignores is dead logic",
    ),
    (
        "malformed_env_disables",
        CLIENT,
        "        logger.warning(\n"
        '            "cursor: ignoring malformed %s=%r (expected a token count such as 250000 '
        'or 250k, or off)",\n'
        "            _ENV_SESSION_TOKEN_BUDGET,\n            raw,\n        )",
        "        return None",
        "a typo in the env var must warn and fall back to the model's window, not silently "
        "switch the bound off",
    ),
    # --- P2: stream health and the shared token-count parser --------------------------
    (
        "close_always_done",
        CLIENT,
        'close = "timeout"',
        'close = "done"',
        "the close reason is the whole point of the line: reporting done for a timeout hides "
        "exactly the failure the instrumentation exists to show",
    ),
    (
        "mode_always_ephemeral",
        CLIENT,
        "mode=mode,\n                close=close,",
        'mode="ephemeral",\n                close=close,',
        "mode is what proves which path ran; a hardcoded label makes the line lie",
    ),
    (
        "session_field_dropped",
        CLIENT,
        'prompt_chars=%d session=%s",',
        'prompt_chars=%d",',
        "without the session id a line cannot be joined to the turn that produced it",
    ),
    (
        "ttfe_sentinel_hidden",
        CLIENT,
        "else -1.0",
        "else 0.0",
        "ttfe=-1.00s must stay distinguishable from a fast first event",
    ),
    (
        "in_tokens_always_zero",
        CLIENT,
        "                    tokens = _context_size_tokens(result, peak_turn_usage)",
        "                    tokens = 0",
        "in_tokens next to prompt_chars is the pair that answers what the provider counts",
    ),
    (
        "logging_guard_removed",
        CLIENT,
        "        except Exception:  # noqa: BLE001 - diagnostics must never break a turn\n"
        "            return\n",
        "",
        "diagnostics must never change a turn's outcome, including a hostile log sink",
    ),
    (
        "k_suffix_dropped",
        CLIENT,
        'if text.endswith("k"):\n            return int(float(text[:-1]) * 1000)',
        'if False:\n            return int(float(text[:-1]) * 1000)',
        "250k in the env override must parse the same way as context=250k",
    ),
    (
        "inferred_always_true",
        TOOL,
        "inferred=bool(last_session)",
        "inferred=True",
        "the summariser must mark an inferred session tag rather than claim it was recorded",
    ),
]


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=str(pathlib.Path(__file__).resolve().parents[1]),
        help="repository root (default: the parent of this script's directory)",
    )
    parser.add_argument("-k", "--only", default="", help="run only mutations whose name contains this")
    args = parser.parse_args(argv)

    root = pathlib.Path(args.repo_root).resolve()
    selected = [m for m in MUTATIONS if args.only in m[0]] if args.only else MUTATIONS
    if not selected:
        print(f"No mutation matches {args.only!r}")
        return 2

    # Back up every file we may touch, byte-for-byte, before touching anything.
    backups: dict[pathlib.Path, pathlib.Path] = {}
    sums: dict[pathlib.Path, str] = {}
    for rel in {m[1] for m in selected}:
        path = root / rel
        if not path.is_file():
            print(f"  ! missing file: {rel} (run from the repository root)")
            return 2
        backup = path.with_suffix(path.suffix + ".mutbak")
        shutil.copy2(path, backup)
        backups[path] = backup
        sums[path] = sha(path)

    caught: list[str] = []
    survived: list[str] = []
    mismatched: list[str] = []

    try:
        for name, rel, old, new, _why in selected:
            path = root / rel
            good = backups[path].read_text(encoding="utf-8")
            occurrences = good.count(old)
            if occurrences != 1:
                mismatched.append(f"{name} (anchor occurs {occurrences}x)")
                print(f"  PATTERN  {name}: anchor occurs {occurrences}x — mutation not applied")
                continue
            mutated = good.replace(old, new, 1)
            if mutated == good:  # paranoia: identical replacement is not a mutation
                mismatched.append(f"{name} (replacement is a no-op)")
                continue
            path.write_text(mutated, encoding="utf-8")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q"],
                    cwd=root,
                    capture_output=True,
                    text=True,
                )
            finally:
                path.write_text(good, encoding="utf-8")
            if result.returncode == 0:
                survived.append(name)
                print(f"  SURVIVED {name}: suite stayed green")
            else:
                caught.append(name)
                print(f"  caught   {name}")
    finally:
        for path, backup in backups.items():
            shutil.copy2(backup, path)
            backup.unlink()

    print()
    restore_ok = True
    for path, want in sums.items():
        got = sha(path)
        ok = got == want
        restore_ok &= ok
        print(f"  restored {path.relative_to(root)}: {'OK' if ok else 'MISMATCH'} ({got[:16]})")

    print()
    if mismatched:
        print("PATTERN ERRORS (fix the anchor, not the tests):")
        for item in mismatched:
            print(f"  - {item}")
    if survived:
        print(f"RESULT: FAIL — {len(survived)}/{len(selected)} mutation(s) survived: "
              f"{', '.join(survived)}")
        print("Each survivor is a regression the suite would not catch.")
    if not restore_ok:
        print("RESULT: FAIL — a source file was not restored to its original bytes")
    if mismatched or survived or not restore_ok:
        return 1
    print(f"RESULT: PASS — {len(caught)}/{len(selected)} mutations caught, sources restored")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
