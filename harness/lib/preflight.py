#!/usr/bin/env python3
"""Cheap preflight: one throwaway profile, lane revision installed, one tiny
Cursor turn. Must exit 0 before the 20 counted runs (Task 7)."""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_one as R  # noqa: E402

PROFILE = "ocg-probe-0"


def main() -> int:
    pdir = R.PROFILES / PROFILE
    key = R.secret()
    rec: dict = {"profile": PROFILE, "lane_sha": R.LANE_SHA, "ok": False}
    if pdir.exists():
        R.run_cmd([R.HERMES, "profile", "delete", PROFILE, "-y"], timeout=180)
    code, out, err, timed = R.run_cmd(
        [R.HERMES, "profile", "create", PROFILE, "--no-alias", "--no-skills"], timeout=300
    )
    if code != 0:
        rec["error"] = f"profile create failed: {R.scrub((out or '') + (err or ''), key)[-1500:]}"
        return finish(rec, pdir)
    R.profile_seed(pdir, stale_override=True)
    ok, log = R.install_plugin(PROFILE)
    rec["install_ok"] = ok
    rec["plugin_sha"] = R.plugin_revision(pdir)
    if not ok or rec["plugin_sha"] != R.LANE_SHA:
        rec["error"] = f"install failed or wrong revision: {log[-1200:]}"
        return finish(rec, pdir)

    env = {k: v for k, v in os.environ.items() if k not in ("GITHUB_TOKEN", "GH_TOKEN")}
    env["CURSOR_API_KEY"] = key
    env["HERMES_CURSOR_MODEL_VARIANTS"] = R.env_value("HERMES_CURSOR_MODEL_VARIANTS") or R.VARIANT
    code, out, err, timed = R.run_cmd(
        [R.HERMES, "-p", PROFILE, "chat", "-q", "Reply with exactly: PRE_OK",
         "--provider", "cursor", "--model", R.VARIANT, "-Q", "--oneshot"],
        env=env, timeout=300,
    )
    rec["exit_code"] = code
    rec["stdout"] = R.scrub((out or "").strip(), key)[-1000:]
    m = re.search(r"^session_id:\s*(\S+)\s*$", err or "", re.MULTILINE)
    rec["session_id"] = m.group(1) if m else ""
    rec["tool_calls"] = R.tool_calls(pdir, rec["session_id"]) if rec["session_id"] else None
    if timed:
        rec["error"] = "preflight timed out"
    elif code != 0:
        rec["error"] = R.scrub((err or out or "").strip(), key)[-1500:]
    rec["ok"] = code == 0 and "PRE_OK" in rec["stdout"] and not timed
    return finish(rec, pdir)


def finish(rec: dict, pdir: Path) -> int:
    (R.HARNESS / "preflight.json").write_text(json.dumps(rec, indent=2) + "\n")
    R.run_cmd([R.HERMES, "profile", "delete", PROFILE, "-y"], timeout=180)
    print(json.dumps({k: rec.get(k) for k in ("ok", "exit_code", "stdout", "plugin_sha", "error")}))
    return 0 if rec.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
