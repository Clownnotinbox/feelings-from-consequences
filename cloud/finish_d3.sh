#!/usr/bin/env bash
# usage: finish_d3.sh HOST OUTDIR — wait for ~/af08/D3_DONE; pull the small pack (decisions + feelings + pilot); verify
# 900 flights (or a pilot-only pack when the stop rule fired); shut the VM down once the pack is safe.
export MSYS_NO_PATHCONV=1; H=$1; OUT=$2; cd "$(dirname "$0")"; mkdir -p "$OUT"
while ! python ssh.py $H run "test -f ~/af08/D3_DONE" 30 > /dev/null 2>&1; do echo "waiting $(date +%H:%M) $(python ssh.py $H run 'tail -n 1 ~/af08/runs/d3.log' 30 2>/dev/null | cut -c1-40)"; sleep 300; done
ok=0; for i in 1 2 3; do python ssh.py $H get /home/ubuntu/af08/runs/d3_small.tgz "$OUT/d3_small.tgz.part" && tar -tzf "$OUT/d3_small.tgz.part" > /dev/null && mv "$OUT/d3_small.tgz.part" "$OUT/d3_small.tgz" && ok=1 && break; sleep 30; done
[ $ok = 1 ] && tar -xzf "$OUT/d3_small.tgz" -C "$OUT"; n=$(ls "$OUT"/stageD3/*.json 2>/dev/null | wc -l); echo "pack ok=$ok flights=$n $(date)"; tail -6 "$OUT/setup.log"
if [ $ok = 1 ] && { [ "$n" -ge 900 ] || grep -q "stopped by the pilot rule" "$OUT/setup.log"; }; then
  python ssh.py $H run "sudo shutdown -h +1" 30 && echo "shutdown requested $H $(date)"
else echo "pack incomplete — VM left running"; exit 2; fi
