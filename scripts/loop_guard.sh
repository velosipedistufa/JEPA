#!/usr/bin/env bash
# loop_guard.sh — keep the MMAGAI autonomous research loop running.
#
# The alarm sound is played ONLY for genuine crashes.
#   rc 0            -> loop finished normally            -> stop, no alarm
#   rc 124          -> 4h timeout wrapper fired          -> stop, no alarm
#   rc 1            -> unhandled python exception        -> loop_v2 already
#                                                          sounded the alarm,
#                                                          just restart
#   rc 130 / 143    -> operator SIGINT / SIGTERM         -> stop, no alarm
#   anything else   -> unexpected death (OOM/SIGKILL...) -> alarm + restart
#
# Usage: scripts/loop_guard.sh
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
LOG="$ROOT/.cycle/loop_v2_run.out"
SOUND="$ROOT/Alarm Sound Effect.m4a"
mkdir -p "$ROOT/.cycle"

play_alarm() {
  if [ -f "$SOUND" ] && command -v ffplay >/dev/null 2>&1; then
    ffplay -nodisp -autoexit -loglevel error "$SOUND" >/dev/null 2>&1 || true
  fi
}

MAX_RESTARTS=5
restarts=0
while :; do
  echo "=== loop_guard: starting loop $(date '+%F %T') ===" >> "$LOG"
  timeout 14400 python -u automation/loop_v2.py --max-cycles 60 --pause 3 >> "$LOG" 2>&1
  rc=$?
  case "$rc" in
    0)
      echo "=== loop_guard: loop finished normally (rc=0) ===" >> "$LOG"; break ;;
    124)
      echo "=== loop_guard: loop hit 4h timeout (rc=124) ===" >> "$LOG"; break ;;
    1)
      echo "=== loop_guard: loop crashed with python exception (rc=1); alarm already sounded, restarting ===" >> "$LOG"
      restarts=$((restarts+1)) ;;
    130|143|129)
      echo "=== loop_guard: loop stopped by signal/operator (rc=$rc) ===" >> "$LOG"; break ;;
    *)
      echo "=== loop_guard: LOOP DIED UNEXPECTEDLY (rc=$rc); sounding alarm ===" >> "$LOG"
      play_alarm
      restarts=$((restarts+1)) ;;
  esac
  if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
    echo "=== loop_guard: too many crashes ($restarts); giving up ===" >> "$LOG"
    break
  fi
  sleep 30
done
