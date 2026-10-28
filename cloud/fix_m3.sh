#!/usr/bin/env bash
# Run as a FILE on the CPU VM: nproc honours OMP_NUM_THREADS=1 (exported just before) -> 0 workers. Use nproc --all.
for p in $(pgrep -f "af08/setup_m.sh"); do kill $p; done; sleep 1
sed -i 's/W=\$(( \$(nproc) - 1 ))/W=$(( $(nproc --all) - 1 ))/' ~/af08/setup_m.sh; grep -o 'W=.*))' ~/af08/setup_m.sh
OMP_NUM_THREADS=1 nproc --all
rm -f ~/af08/M_DONE ~/af08/runs/CLOSED_LOOP_v4_m*.json; echo "restart with workers fix $(date)" >> ~/af08/runs/setup.log
(setsid nohup bash ~/af08/setup_m.sh > /dev/null 2>&1 < /dev/null &); sleep 2; echo restarted
