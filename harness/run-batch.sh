#!/usr/bin/env bash
# The 20-run bar: 10 fold + 10 audit on the plugin arm, interleaved.
# Safe to re-run: each invocation continues from the JSONs already on disk.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PER_BRIEF="${PER_BRIEF:-10}"
BATCH_LOG="$HERE/logs/batch.log"
mkdir -p "$HERE/runs" "$HERE/logs"

count_done() {  # $1=brief
  python3 - "$HERE/runs" "$1" <<'PY'
import json, sys
from pathlib import Path
runs, brief = Path(sys.argv[1]), sys.argv[2]
n = 0
for p in runs.glob("*.json"):
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    if d.get("brief") == brief and d.get("arm") == "plugin":
        n += 1
print(n)
PY
}

has_no_override() {  # $1=brief
  python3 - "$HERE/runs" "$1" <<'PY'
import json, sys
from pathlib import Path
runs, brief = Path(sys.argv[1]), sys.argv[2]
for p in runs.glob("*.json"):
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    if d.get("brief") == brief and d.get("stale_override") is False:
        print("yes"); raise SystemExit
print("no")
PY
}

echo "[$(date -Is)] batch start (target ${PER_BRIEF} per brief)" | tee -a "$BATCH_LOG"
for ((i = 1; i <= PER_BRIEF; i++)); do
  for brief in fold audit; do
    done_n="$(count_done "$brief")"
    if (( done_n >= PER_BRIEF )); then
      continue
    fi
    override=1
    if [[ "$(has_no_override "$brief")" == "no" ]]; then
      override=0   # first counted run of this brief exercises the no-override path
    fi
    echo "[$(date -Is)] start $brief #$((done_n + 1)) override=$override" | tee -a "$BATCH_LOG"
    HARNESS_STALE_OVERRIDE="$override" "$HERE/run-one.sh" "$brief" plugin >>"$BATCH_LOG" 2>&1
    rc=$?
    echo "[$(date -Is)] end $brief rc=$rc" | tee -a "$BATCH_LOG"
  done
done
echo "[$(date -Is)] batch done" | tee -a "$BATCH_LOG"
