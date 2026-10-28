#!/usr/bin/env bash
# usage: finish_job.sh HOST OUTDIR DONE_MARKER CHECK_PY — every 5 min pull ~/af08/runs (json + logs) into OUTDIR; when
# ~/af08/DONE_MARKER exists and LAST_MARKER says DONE_MARKER, pull once more, run CHECK_PY OUTDIR (exit 0 = complete) and
# only then shut the VM down (it is set to delete after shutdown). One finisher per host (token file).
export MSYS_NO_PATHCONV=1; H=$1; OUT=$2; M=$3; CHK=$4; cd "$(dirname "$0")"; mkdir -p "$OUT"
TOK="finish-job-$H.token"; ME="$$-$RANDOM-$(date +%s)"; echo "$ME" > "$TOK"; mine() { [ "$(cat "$TOK" 2>/dev/null)" = "$ME" ]; }
pull() { python ssh.py $H run "cd ~/af08/runs && tar czf /tmp/r.tgz --ignore-failed-read *.json *.log > /dev/null 2>&1" 120 > /dev/null 2>&1 && python ssh.py $H get /tmp/r.tgz "$OUT/r.tgz.part" > /dev/null 2>&1 && tar -tzf "$OUT/r.tgz.part" > /dev/null 2>&1 && mv "$OUT/r.tgz.part" "$OUT/r.tgz" && tar -xzf "$OUT/r.tgz" -C "$OUT" && return 0; return 1; }
fails=0
while true; do
  mine || { echo "retired $(date)"; exit 0; }
  if pull; then fails=0; echo "pulled $(date +%H:%M)"; else fails=$((fails+1)); echo "pull failed ($fails) $(date +%H:%M)"; fi
  [ $fails -ge 6 ] && { echo "VM unreachable for 30 min, stopping finisher $(date)"; exit 1; }
  if python ssh.py $H run "test -f ~/af08/$M" 30 > /dev/null 2>&1; then
    pull; last=$(python ssh.py $H run "cat ~/af08/LAST_MARKER" 30 2>/dev/null)
    if [ "$last" = "$M" ] && python "$CHK" "$OUT"; then python ssh.py $H run "sudo shutdown -h +1" 30 && echo "complete; shutdown requested $H $(date)"; exit 0; fi
    echo "DONE marker but check failed (last=$last) — VM left running for inspection $(date)"; exit 2
  fi
  sleep 300
done
