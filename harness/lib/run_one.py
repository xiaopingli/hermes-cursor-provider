#!/usr/bin/env python3
"""One Task 7 acceptance run: seed a throwaway profile, install the lane revision,
run one frozen brief on the plugin arm, and write runs/<id>.json.

Spec: Mission Control 01_plans/hermes-plans/cursor-provider-native-tools.md, Task 7.
Never prints credentials. Writes only under this harness dir, the throwaway
profile, and the throwaway clone.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
# The harness runs from two layouts: beside the fork (<state>/harness + <state>/fork, the
# original dev layout) or inside the fork repository (harness/ at the repo root). Resolve the
# fork explicitly rather than assuming the parent, so neither layout is privileged.
_FORK_ENV = os.environ.get("HARNESS_FORK", "").strip()
if _FORK_ENV:
    FORK = Path(_FORK_ENV).expanduser().resolve()
elif (HARNESS.parent / "fork" / "provider").is_dir():
    FORK = HARNESS.parent / "fork"
else:
    FORK = HARNESS.parent
STATE = FORK.parent
LANE_SHA = os.environ.get(
    "HARNESS_LANE_SHA", "dcb8812a0065bcd4aa031f0a9ec3dc495664fe4e"
)
VARIANT = "grok-4.7@context=500k,reasoning_effort=xhigh,fast=true"
# The lane clones a small fixture repo. In the repository the fixture ships as a git bundle
# (which preserves the base SHA); a checked-out fixtures/tinyrepo wins when one exists.
_FIXTURE_DIR = HARNESS / "fixtures" / "tinyrepo"
SRC_REPO = (
    _FIXTURE_DIR
    if _FIXTURE_DIR.is_dir()
    else HARNESS / "fixtures" / "tinyrepo.bundle"
)
BASE_SHA = "95f5fd6a3a8dcbcb67453d781bace5b7d8db47d8"
HERMES_HOME = Path(
    os.environ.get("HERMES_HOME", "").strip() or (Path.home() / ".hermes")
).expanduser()
ROOT_ENV = Path(
    os.environ.get("HARNESS_ROOT_ENV", "").strip() or (HERMES_HOME / ".env")
).expanduser()
PROFILES = HERMES_HOME / "profiles"
RUNS = HARNESS / "runs"
LOGS = HARNESS / "logs"
CLONES = HARNESS / "clones"
BRIEFS = HARNESS / "briefs"
HERMES = shutil.which("hermes") or str(
    HERMES_HOME.parent / ".local" / "bin" / "hermes"
)
RUN_TIMEOUT = int(os.environ.get("HARNESS_RUN_TIMEOUT", "3600"))
TEST_TIMEOUT = int(os.environ.get("HARNESS_TEST_TIMEOUT", "600"))


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def secret() -> str:
    """Read CURSOR_API_KEY from the root .env without ever printing it."""
    val = ""
    if ROOT_ENV.exists():
        for line in ROOT_ENV.read_text(errors="replace").splitlines():
            if line.startswith("CURSOR_API_KEY="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not val:
        val = os.environ.get("CURSOR_API_KEY", "")
    if not val:
        raise SystemExit("CURSOR_API_KEY is not configured (root .env or environment)")
    return val


def env_value(key: str) -> str:
    if ROOT_ENV.exists():
        for line in ROOT_ENV.read_text(errors="replace").splitlines():
            if line.startswith(key + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get(key, "")


def scrub(text: str, key: str) -> str:
    if not text:
        return ""
    if key:
        text = text.replace(key, "[REDACTED]")
    return text


def run_cmd(cmd, *, cwd=None, env=None, timeout=None, capture=True):
    """Run a command, kill its whole process group on timeout."""
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        start_new_session=True,
        text=True,
        errors="replace",
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out or "", err or "", False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, 9)
        except OSError:
            pass
        out, err = proc.communicate()
        return None, out or "", err or "", True


def allocate(brief: str, arm: str) -> tuple[int, str]:
    """Atomically allocate the next global run number."""
    RUNS.mkdir(parents=True, exist_ok=True)
    seq_path = RUNS / ".seq"
    with open(seq_path, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        raw = fh.read().strip()
        n = int(raw) + 1 if raw else 1
        fh.seek(0)
        fh.truncate()
        fh.write(str(n))
        fh.flush()
        os.fsync(fh.fileno())
        fcntl.flock(fh, fcntl.LOCK_UN)
    return n, f"{brief}-{arm}-{n:02d}"


def profile_seed(profile_dir: Path, stale_override: bool) -> dict:
    """Copy the two root-only settings the bar exercises into the profile.

    CURSOR_API_KEY is NOT written to disk; it is exported into the child env.
    """
    variants = env_value("HERMES_CURSOR_MODEL_VARIANTS") or VARIANT
    envp = profile_dir / ".env"
    text = envp.read_text(errors="replace") if envp.exists() else ""
    if "HERMES_CURSOR_MODEL_VARIANTS=" not in text:
        if not text.endswith("\n"):
            text += "\n"
        text += (
            "\n# harness seed (Task 7): parameterised variant listing\n"
            f"HERMES_CURSOR_MODEL_VARIANTS={variants}\n"
        )
        envp.write_text(text)
    cfgp = profile_dir / "config.yaml"
    cfg = cfgp.read_text(errors="replace") if cfgp.exists() else ""
    if "model_overrides:" not in cfg:
        if not cfg.endswith("\n"):
            cfg += "\n"
        cfg += (
            "\n# harness seed (Task 7): context window for the exact variant\n"
            "model_overrides:\n"
            "  cursor:\n"
            "    grok-4.7@context=500k,reasoning_effort=xhigh,fast=true:\n"
            "      context_window: 500000\n"
        )
    stale_key = "stale_timeout_seconds"
    if stale_override and stale_key not in cfg:
        # insert into the existing top-level providers: block (keeps the rest verbatim)
        lines = cfg.splitlines()
        inserted = False
        for i, line in enumerate(lines):
            if line.rstrip() == "providers:":
                lines[i + 1 : i + 1] = [
                    "  cursor:",
                    "    stale_timeout_seconds: 900",
                ]
                inserted = True
                break
        if not inserted:
            lines += ["providers:", "  cursor:", "    stale_timeout_seconds: 900"]
        cfg = "\n".join(lines) + "\n"
    cfgp.write_text(cfg)
    applied = stale_key in cfgp.read_text(errors="replace")
    return {"variants": variants == env_value("HERMES_CURSOR_MODEL_VARIANTS") or bool(variants),
            "stale_override_applied": applied}


def install_plugin(profile: str) -> tuple[bool, str]:
    cmd = [
        HERMES, "-p", profile, "plugins", "install",
        f"file://{FORK}#provider",
        "--ref", LANE_SHA,
        "--force", "--yes-deps", "--enable",
    ]
    code, out, err, timed = run_cmd(cmd, timeout=600)
    detail = scrub((out or "") + (err or ""), secret())
    return code == 0 and not timed, detail[-3000:]


def plugin_revision(profile_dir: Path) -> str | None:
    p = profile_dir / "plugins" / ".install-metadata.json"
    try:
        data = json.loads(p.read_text())
        rec = data.get("cursor-provider") or {}
        rev = rec.get("revision")
        return rev if isinstance(rev, str) else None
    except Exception:
        return None


def make_clone(run_id: str, branch: str | None) -> Path:
    CLONES.mkdir(parents=True, exist_ok=True)
    dest = CLONES / run_id
    if dest.exists():
        shutil.rmtree(dest)
    code, out, err, _ = run_cmd(["git", "clone", "--quiet", str(SRC_REPO), str(dest)], timeout=600)
    if code != 0:
        raise RuntimeError(f"git clone failed: {scrub(err, '')[-800:]}")
    if branch:
        code, out, err, _ = run_cmd(
            ["git", "-C", str(dest), "checkout", "-q", "-B", branch, BASE_SHA], timeout=120
        )
    else:
        code, out, err, _ = run_cmd(
            ["git", "-C", str(dest), "checkout", "-q", "--detach", BASE_SHA], timeout=120
        )
    if code != 0:
        raise RuntimeError(f"git checkout failed: {scrub(err, '')[-800:]}")
    run_cmd(["git", "-C", str(dest), "remote", "remove", "origin"], timeout=60)
    code, out, err, _ = run_cmd(["git", "-C", str(dest), "remote", "-v"], timeout=60)
    if (out or "").strip():
        raise RuntimeError("clone still has a remote after `git remote remove origin`")
    # Per-run global git config: no ~/.gitconfig, no ~/.git-credentials.
    gitcfg = dest.parent / f"{run_id}.gitconfig"
    gitcfg.write_text(
        "[user]\n\tname = Hermes Harness\n\temail = harness@example.invalid\n"
        "[credential]\n\thelper =\n"
        "[init]\n\tdefaultBranch = main\n"
    )
    return dest


def log_snapshot(path: Path) -> dict:
    try:
        st = path.stat()
        return {"exists": True, "inode": st.st_ino, "size": st.st_size}
    except FileNotFoundError:
        return {"exists": False, "inode": None, "size": 0}


def _read_bytes(path: Path) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def derive_max_calls_per_turn(text: str) -> int:
    """Max Hermes tool executions between consecutive API calls, per the patch plan's §6 item 4.

    Counts `tool ... completed`, `tool ... failed` and `Tool ... returned error` between
    `agent.conversation_loop: API call #N` lines, plus the tail block after the last call.
    A FLOOR: it counts Hermes-executed calls for one assistant message, while the plugin's
    cap also counts the nudge's second run and undelivered sink entries.
    """
    api_call = re.compile(r"agent\.conversation_loop: API call #\d+")
    tool_exec = re.compile(
        r"agent\.tool_executor: (?:tool \S+ (?:completed|failed)|Tool \S+ returned error)"
    )
    blocks: list[int] = []
    current = None
    for line in text.splitlines():
        if api_call.search(line):
            if current is not None:
                blocks.append(current)
            current = 0
        elif current is not None and tool_exec.search(line):
            current += 1
    if current is not None:
        blocks.append(current)
    return max(blocks) if blocks else 0


def max_reported_input_tokens(text: str) -> int:
    """The largest `in=` reported by any API call in this run, or 0 when none parsed.

    P1's stop gate reads this: a bounded session must show a maximum reported input at or
    under the budget, measured rather than asserted. The value comes from the same log slice
    the PASS rule uses, so it covers exactly this run's window.
    """
    best = 0
    for match in re.finditer(r"agent\.conversation_loop: API call #\d+:[^\n]*?\bin=(\d+)", text):
        best = max(best, int(match.group(1)))
    return best


def log_slice(log_path: Path, snap: dict) -> dict:
    """Two-segment slice across a possible mid-run rollover; unverifiable => fail."""
    try:
        end = log_path.stat()
    except FileNotFoundError:
        if not snap["exists"]:
            return {"verified": True, "text": "", "note": "no agent.log written"}
        return {"verified": False, "text": "", "note": "agent.log vanished during the run"}
    try:
        if not snap["exists"]:
            if log_path.with_name("agent.log.1").exists():
                return {"verified": False, "text": "",
                        "note": "no start snapshot and agent.log.1 exists (rollover unprovable)"}
            data = _read_bytes(log_path)[: end.st_size]
        elif snap["inode"] == end.st_ino:
            if end.st_size < snap["size"]:
                return {"verified": False, "text": "",
                        "note": "agent.log truncated in place during the run"}
            data = _read_bytes(log_path)[snap["size"]: end.st_size]
        else:
            rot = log_path.with_name("agent.log.1")
            if not rot.exists():
                return {"verified": False, "text": "",
                        "note": "inode changed but agent.log.1 is missing"}
            rst = rot.stat()
            if rst.st_ino != snap["inode"]:
                return {"verified": False, "text": "",
                        "note": "agent.log.1 inode does not match the start snapshot"}
            two = log_path.with_name("agent.log.2")
            if two.exists() and two.stat().st_ino == snap["inode"]:
                return {"verified": False, "text": "",
                        "note": "more than one rollover is evident"}
            seg1 = _read_bytes(rot)[snap["size"]:]
            seg2 = _read_bytes(log_path)[: end.st_size]
            data = seg1 + seg2
        return {"verified": True, "text": data.decode("utf-8", "replace"), "note": ""}
    except OSError as exc:
        return {"verified": False, "text": "", "note": f"log read failed: {exc}"}


def tool_calls(profile_dir: Path, session_id: str) -> int | None:
    db = profile_dir / "state.db"
    if not db.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)
        try:
            row = con.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id=? AND role='tool'",
                (session_id,),
            ).fetchone()
            return int(row[0]) if row else 0
        finally:
            con.close()
    except Exception:
        return None


def latest_session(profile_dir: Path) -> str:
    """Recovery only: a killed run never prints its session_id line."""
    db = profile_dir / "state.db"
    if not db.exists():
        return ""
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)
        try:
            row = con.execute(
                "SELECT id FROM sessions ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            return str(row[0]) if row else ""
        finally:
            con.close()
    except Exception:
        return ""


def git_lines(clone: Path, *args: str) -> list[str]:
    code, out, _, _ = run_cmd(["git", "-C", str(clone), *args], timeout=120)
    if code != 0:
        return []
    return [ln for ln in out.splitlines() if ln.strip()]


def render_brief(brief: str, clone: Path, dest: Path) -> str:
    src = BRIEFS / f"{brief}.frozen.md"
    text = src.read_text()
    text = text.replace("{{CLONE}}", str(clone))
    text = text.replace("{{HARNESS}}", str(HARNESS))
    text = text.replace("{{HOME}}", str(Path.home()))
    dest.write_text(text)
    return text


def artifact_fold(clone: Path, env: dict) -> dict:
    commits = git_lines(clone, "log", "--oneline", "--no-decorate", f"{BASE_SHA}..HEAD")
    # The fixture repo is stdlib-only: use the system interpreter and drop the
    # parent's Python path so an inherited Hermes venv cannot shadow pytest.
    test_env = {k: v for k, v in env.items()
                if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    test_env["PYTHONDONTWRITEBYTECODE"] = "1"
    code, out, err, timed = run_cmd(
        ["/usr/bin/python3", "-m", "pytest", "-q"], cwd=clone, env=test_env,
        timeout=TEST_TIMEOUT,
    )
    tail = scrub(((out or "") + (err or ""))[-4000:], env.get("CURSOR_API_KEY", ""))
    return {
        "kind": "fold",
        "commit_count": len(commits),
        "commits": commits,
        "test_command": "python3 -m pytest -q",
        "test_exit": code,
        "test_timeout": timed,
        "test_output": tail,
        "present": bool(commits) and bool(tail.strip()),
    }


def artifact_audit(stdout: str, stderr: str) -> dict:
    text = (stdout or "") + "\n" + (stderr or "")
    matches = re.findall(r"^SCORE:\s*(\d{1,3})/100\s*$", text, re.MULTILINE)
    line = f"SCORE: {matches[-1]}/100" if matches else ""
    return {"kind": "audit", "score": int(matches[-1]) if matches else None,
            "score_line": line, "present": bool(matches)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("brief", choices=["fold", "audit"])
    ap.add_argument("arm", choices=["plugin"])
    args = ap.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    n, run_id = allocate(args.brief, args.arm)
    profile = f"ocg-probe-{n}"
    pdir = PROFILES / profile
    started = now_iso()
    t0 = time.monotonic()
    stale_env = os.environ.get("HARNESS_STALE_OVERRIDE", "1") != "0"

    rec: dict = {
        "id": run_id, "brief": args.brief, "arm": args.arm, "profile": profile,
        "lane_sha": LANE_SHA, "started_at": started, "stale_override_requested": stale_env,
        "exit_code": None, "error": "", "tool_calls": None, "commits": [],
        "artifact": {}, "plugin_sha": None, "plan_only": None,
        "stale_override": None, "kill_retry_marker": None, "kill_retry_verified": None,
        "run_timeout_raised": False, "pass": False, "fail_reasons": [],
    }
    key = secret()
    child_env = {k: v for k, v in os.environ.items()
                 if k not in ("GITHUB_TOKEN", "GH_TOKEN", "GIT_DIR", "GIT_WORK_TREE",
                              "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY")}
    child_env["CURSOR_API_KEY"] = key
    child_env["HERMES_CURSOR_MODEL_VARIANTS"] = env_value("HERMES_CURSOR_MODEL_VARIANTS") or VARIANT
    child_env["GIT_TERMINAL_PROMPT"] = "0"

    def finish(rec: dict, clone: Path | None, keep_clone: bool) -> int:
        rec["ended_at"] = now_iso()
        rec["duration_s"] = round(time.monotonic() - t0, 1)
        try:  # keep the profile's logs after the profile itself is deleted
            for name in ("agent.log", "errors.log"):
                src = pdir / "logs" / name
                if src.exists():
                    shutil.copy2(src, LOGS / f"{run_id}.{name}")
        except OSError:
            pass
        tmp = RUNS / f".{run_id}.json.tmp"
        tmp.write_text(json.dumps(rec, indent=2) + "\n")
        os.replace(tmp, RUNS / f"{run_id}.json")
        if profile:
            run_cmd([HERMES, "profile", "delete", profile, "-y"], timeout=180)
        if clone is not None and not keep_clone and clone.exists():
            shutil.rmtree(clone, ignore_errors=True)
            rec["clone_deleted"] = True
        print(json.dumps({k: rec.get(k) for k in
                          ("id", "exit_code", "tool_calls", "plan_only", "plugin_sha",
                           "stale_override", "kill_retry_marker", "run_timeout_raised",
                           "pass", "fail_reasons")}))
        return 0 if rec.get("pass") else 1

    # --- load path ---------------------------------------------------------
    if pdir.exists():
        run_cmd([HERMES, "profile", "delete", profile, "-y"], timeout=180)
    code, out, err, timed = run_cmd(
        [HERMES, "profile", "create", profile, "--no-alias", "--no-skills"], timeout=300
    )
    if code != 0 or timed:
        rec["error"] = f"profile create failed: {scrub((out or '') + (err or ''), key)[-1500:]}"
        rec["fail_reasons"] = ["profile_create"]
        return finish(rec, None, False)
    seed = profile_seed(pdir, stale_env)
    rec["stale_override"] = bool(seed["stale_override_applied"])
    if stale_env and not rec["stale_override"]:
        rec["error"] = "stale override requested but not present in the seeded config"
        rec["fail_reasons"] = ["stale_seed"]
        return finish(rec, None, False)

    ok, install_log = install_plugin(profile)
    if not ok:
        rec["error"] = f"plugin install failed: {install_log[-1500:]}"
        rec["fail_reasons"] = ["install"]
        return finish(rec, None, False)
    rec["plugin_sha"] = plugin_revision(pdir)
    if rec["plugin_sha"] != LANE_SHA:
        rec["error"] = f"installed revision {rec['plugin_sha']!r} != lane {LANE_SHA}"
        rec["fail_reasons"] = ["plugin_sha"]
        return finish(rec, None, False)

    # --- throwaway clone + frozen brief ------------------------------------
    clone = None
    try:
        clone = make_clone(run_id, "feat/windows" if args.brief == "fold" else None)
    except Exception as exc:  # noqa: BLE001
        rec["error"] = str(exc)[-1500:]
        rec["fail_reasons"] = ["clone"]
        return finish(rec, clone, True)
    rec["clone"] = str(clone)
    rec["clone_head_before"] = (git_lines(clone, "rev-parse", "HEAD") or [""])[0]
    child_env["GIT_CONFIG_GLOBAL"] = str(clone.parent / f"{run_id}.gitconfig")
    brief_path = RUNS / f"{run_id}.brief.md"
    render_brief(args.brief, clone, brief_path)
    rec["brief_path"] = str(brief_path)

    # --- run ---------------------------------------------------------------
    log_path = pdir / "logs" / "agent.log"
    snap = log_snapshot(log_path)
    cmd = [HERMES, "-p", profile, "chat", "--query-file", str(brief_path),
           "--provider", "cursor", "--model", VARIANT, "-Q", "--oneshot"]
    code, out, err, timed = run_cmd(cmd, cwd=clone, env=child_env, timeout=RUN_TIMEOUT)
    rec["exit_code"] = code
    (LOGS / f"{run_id}.out.txt").write_text(scrub(out or "", key))
    (LOGS / f"{run_id}.err.txt").write_text(scrub(err or "", key))
    if timed:
        rec["error"] = f"run exceeded HARNESS_RUN_TIMEOUT={RUN_TIMEOUT}s and was killed"
    elif code != 0:
        rec["error"] = scrub((err or out or "").strip(), key)[-2000:]
    # Bounded-wait evidence: true only when the plugin's cursor_run_timeout is
    # what failed the run (status 408), not an outer harness kill.
    combined = "\n".join((out or "", err or "", rec["error"]))
    rec["run_timeout_raised"] = bool(
        rec["exit_code"] != 0
        and ("cursor_run_timeout" in combined
             or "Cursor SDK run did not finish within" in combined)
    )

    sid = ""
    m = re.search(r"^session_id:\s*(\S+)\s*$", err or "", re.MULTILINE)
    if m:
        sid = m.group(1)
    if not sid:
        # Killed/timed-out runs never print the stderr line; the profile holds
        # exactly one session, so recover it and mark the provenance.
        sid = latest_session(pdir)
        if sid:
            rec["session_id_recovered"] = True
    rec["session_id"] = sid
    if sid:
        tc = tool_calls(pdir, sid)
        rec["tool_calls"] = tc
        rec["plan_only"] = (tc == 0) if tc is not None else None
    else:
        rec["tool_calls"] = None
        rec["plan_only"] = None

    # --- evidence ----------------------------------------------------------
    rec["commits"] = git_lines(clone, "log", "--oneline", "--no-decorate", f"{BASE_SHA}..HEAD")
    rec["head_after"] = (git_lines(clone, "rev-parse", "HEAD") or [""])[0]
    if args.brief == "fold":
        rec["artifact"] = artifact_fold(clone, child_env)
    else:
        rec["artifact"] = artifact_audit(out or "", err or "")

    sl = log_slice(log_path, snap)
    rec["kill_retry_verified"] = sl["verified"]
    rec["kill_retry_marker"] = ("Stream stale for" in sl["text"]) if sl["verified"] else None
    rec["log_slice_note"] = sl["note"]
    # PASS-rule inputs: both gate the verdict below. "turn limit reached" is the plugin's
    # limit-refusal log line; "refusing tool call" would also match the no-id refusal.
    rec["cap_refusals"] = sl["text"].count("turn limit reached")
    rec["duplicate_tool_call_warnings"] = sl["text"].count("Removed duplicate tool call")
    # §6 item 4 of the patch plan: record the per-turn call high-water mark so the 64-call
    # constant is justified by measurement rather than argument. A FLOOR (see the helper).
    rec["max_distinct_calls_per_turn"] = derive_max_calls_per_turn(sl["text"])
    # P1's stop gate: the largest `in=` the run actually submitted. Bounded sessions are
    # judged on this number, so it must be recorded rather than eyeballed.
    rec["max_reported_input_tokens"] = max_reported_input_tokens(sl["text"])
    # Diagnostics only (not part of the PASS rule): the signature that preceded every
    # observed hang.
    rec["nonstream_stale_marker"] = "Non-streaming API call stale for" in sl["text"]
    if not sl["verified"]:
        rec["fail_reasons"].append("kill_retry_unverifiable")

    # --- verdict -----------------------------------------------------------
    fails = rec["fail_reasons"]
    if rec["exit_code"] != 0:
        fails.append("exit_code")
    if rec["error"]:
        fails.append("error")
    if not isinstance(rec["tool_calls"], int) or rec["tool_calls"] < 1:
        fails.append("tool_calls")
    if not rec["artifact"].get("present"):
        fails.append("artifact")
    if rec["plan_only"] is not False:
        fails.append("plan_only")
    if rec["kill_retry_marker"] is not False:
        fails.append("kill_retry_marker")
    if rec["plugin_sha"] != LANE_SHA:
        fails.append("plugin_sha")
    if rec["cap_refusals"]:
        fails.append("cap_refusals")
    if rec["duplicate_tool_call_warnings"]:
        fails.append("duplicate_tool_calls")
    rec["fail_reasons"] = sorted(set(fails))
    rec["pass"] = not rec["fail_reasons"]
    return finish(rec, clone, keep_clone=not rec["pass"])


if __name__ == "__main__":
    sys.exit(main())
