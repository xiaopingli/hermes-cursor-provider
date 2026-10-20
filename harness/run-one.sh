#!/usr/bin/env bash
# Task 7 harness entry point: harness/run-one.sh <brief> <arm>
#   brief: fold | audit
#   arm:   plugin (the gate arm; the new plugin runs inside Hermes)
# Env: HARNESS_STALE_OVERRIDE=0 omits providers.cursor.stale_timeout_seconds
#      (run-batch.sh sets that on the first run of each brief).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -ne 2 ]]; then
  echo "usage: $0 <fold|audit> <plugin>" >&2
  exit 2
fi
exec /usr/bin/env python3 "$HERE/lib/run_one.py" "$1" "$2"
