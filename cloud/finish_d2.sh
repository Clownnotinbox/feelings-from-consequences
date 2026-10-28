#!/usr/bin/env bash
# usage: finish_d2.sh HOST OUTDIR — wait for ~/af08/D2_DONE; pull the small pack (decisions + feelings, verified: 900
# flights), then try the frames (npz, large) for 20 min; shut the VM down once the small pack is safe.
export MSYS_NO_PATHCONV=1; H=$1; OUT=$2; cd "$(dirname "$0")"; mkdir -p "$OUT"
while ! python ssh.py $H run "test -f ~/af08/D2_DONE" 30 > /dev/null 2>&1; do echo "waiting $(date +%H:%M) $(python ssh.py $H run 'tail -n 1 ~/af08/runs/d2.log' 30 2>/dev/null | cut -c1-60)"; sleep 300; done
ok=0; for i in 1 2 3; do python ssh.py $H get /home/ubuntu/af08/runs/d2_small.tgz "$OUT/d2_small.tgz.part" && tar -tzf "$OUT/d2_small.tgz.part" > /dev/null && mv "$OUT/d2_small.tgz.part" "$OUT/d2_small.tgz" && ok=1 && break; sleep 30; done
[ $ok = 1 ] && tar -xzf "$OUT/d2_small.tgz" -C "$OUT"; n=$(ls "$OUT"/stageD2/*.json 2>/dev/null | wc -l); echo "small ok=$ok flights=$n $(date)"
[ $ok = 1 ] && [ "$n" -ge 900 ] || { echo "small pack incomplete — VM left running"; exit 2; }
timeout 1200 python ssh.py $H get /home/ubuntu/af08/runs/d2_npz.tar "$OUT/d2_npz.tar.part" && tar -tf "$OUT/d2_npz.tar.part" > /dev/null && mv "$OUT/d2_npz.tar.part" "$OUT/d2_npz.tar" && echo "npz ok $(date)" || echo "npz not pulled (frames stay unavailable) $(date)"
python ssh.py $H run "sudo shutdown -h +1" 30 && echo "shutdown requested $H $(date)"
