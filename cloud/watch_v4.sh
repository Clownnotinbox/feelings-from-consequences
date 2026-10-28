#!/usr/bin/env bash
# Exits (= one notification) when the v4 VM needs attention: done, a Python error, no progress for 30 min, or SSH lost.
export MSYS_NO_PATHCONV=1; H=$1; cd "$(dirname "$0")"; fail=0; last=-1; stall=0
while true; do
  r=$(python ssh.py $H run "ls ~/af08/runs/stageA4/*.npz 2>/dev/null | wc -l; grep -c Traceback ~/af08/runs/v4.log; test -f ~/af08/V4_DONE && echo DONE" 30 2>/dev/null)
  if [ -z "$r" ]; then fail=$((fail+1)); [ $fail -ge 3 ] && { echo "SSH lost $(date)"; exit 1; }; sleep 300; continue; fi
  fail=0; n=$(echo "$r" | sed -n 1p); tb=$(echo "$r" | sed -n 2p)
  echo "$r" | grep -q DONE && { echo "V4 DONE n=$n $(date)"; exit 0; }
  [ "${tb:-0}" -gt 0 ] && { echo "Traceback in v4.log n=$n $(date)"; exit 2; }
  if [ "$n" = "$last" ]; then stall=$((stall+1)); [ $stall -ge 6 ] && { echo "no progress 30 min n=$n $(date)"; exit 3; }; else stall=0; last=$n; fi
  sleep 300
done
