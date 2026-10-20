#!/usr/bin/env bash
# One brief's lane: keep running until TARGET *passing* runs exist at the lane revision.
# Usage: run-lane.sh <fold|audit> [target] [auto|0|1]
#   auto = runs get no staleness override until one no-override run has PASSED, bounded by
#          NO_OVERRIDE_MAX_ATTEMPTS no-override runs recorded at the lane sha; the rest get 900s.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIEF="${1:?brief}"; TARGET="${2:-10}"; MODE="${3:-auto}"
# Only count runs at the revision under test; older evidence stays on disk.
LANE_SHA="dcb8812a0065bcd4aa031f0a9ec3dc495664fe4e"
mkdir -p "$HERE/runs" "$HERE/logs"
exec 9>"$HERE/logs/lane-$BRIEF.lock"
if ! flock -n 9; then echo "lane $BRIEF already running" >&2; exit 1; fi
LOG="$HERE/logs/lane-$BRIEF.log"

# Quota guard, MODE=auto only (MODE=0 never falls back; MODE=1 never enters the auto branch).
# It counts no-override runs RECORDED AT LANE_SHA in runs/*.json, pass or fail: the same source
# the counters below read, so the count persists across invocations and a restart does not
# restore a spent allowance. Once spent the lane runs override-only, and the done line reports
# EVIDENCE MISSING unless a no-override run passed. Lifting this bound for a second pass, or
# recording the weaker evidence, is Li's decision, not this script's.
NO_OVERRIDE_MAX_ATTEMPTS=3

count_done() {
  python3 - "$HERE/runs" "$BRIEF" "$LANE_SHA" <<'PY'
import json, sys
from pathlib import Path
runs, brief, lane = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
n = 0
for p in runs.glob("*.json"):
    try: d = json.loads(p.read_text())
    except Exception: continue
    if d.get("brief") == brief and d.get("arm") == "plugin" and d.get("lane_sha") == lane and d.get("pass") is True: n += 1
print(n)
PY
}

has_no_override() {
  python3 - "$HERE/runs" "$BRIEF" "$LANE_SHA" <<'PY'
import json, sys
from pathlib import Path
runs, brief, lane = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
for p in runs.glob("*.json"):
    try: d = json.loads(p.read_text())
    except Exception: continue
    if d.get("brief") == brief and d.get("lane_sha") == lane and d.get("stale_override") is False and d.get("pass") is True:
        print("yes"); raise SystemExit
print("no")
PY
}

# $1 = "pass": passing no-override runs (the evidence). $1 = "any": every no-override run
# recorded (the NO_OVERRIDE_MAX_ATTEMPTS basis).
count_no_override() {
  python3 - "$HERE/runs" "$BRIEF" "$LANE_SHA" "$1" <<'PY'
import json, sys
from pathlib import Path
runs, brief, lane, which = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
n = 0
for p in runs.glob("*.json"):
    try: d = json.loads(p.read_text())
    except Exception: continue
    if d.get("brief") != brief or d.get("lane_sha") != lane or d.get("stale_override") is not False: continue
    if which == "pass" and d.get("pass") is not True: continue
    n += 1
print(n)
PY
}

echo "[$(date -Is)] lane $BRIEF start target=$TARGET mode=$MODE" | tee -a "$LOG"
while :; do
  done_n="$(count_done)"
  (( done_n >= TARGET )) && break
  override=1
  if [[ "$MODE" == "auto" && "$(has_no_override)" == "no" ]] \
      && (( $(count_no_override any) < NO_OVERRIDE_MAX_ATTEMPTS )); then
    override=0
  fi
  if [[ "$MODE" == "0" ]]; then override=0; fi
  echo "[$(date -Is)] lane $BRIEF run #$((done_n + 1)) override=$override" | tee -a "$LOG"
  HARNESS_STALE_OVERRIDE="$override" "$HERE/run-one.sh" "$BRIEF" plugin >>"$LOG" 2>&1
  rc=$?
  echo "[$(date -Is)] lane $BRIEF run rc=$rc" | tee -a "$LOG"
done
no_override_passes="$(count_no_override pass)"
echo "[$(date -Is)] lane $BRIEF done ($(count_done)/$TARGET) no-override passes: $no_override_passes" | tee -a "$LOG"
if (( no_override_passes == 0 )); then
  echo "[$(date -Is)] lane $BRIEF EVIDENCE MISSING: no passing no-override run at $LANE_SHA, so plan section 6 item 3 is NOT met. Lift the no-override bound for a second pass, or record the weaker evidence as Li's decision." | tee -a "$LOG"
fi
