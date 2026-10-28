#!/usr/bin/env bash
# Restart stage D and the queued closed loop with the fixed pilot. Run as a FILE (its own command line contains none
# of the process names it kills).
pkill -f run_d_after_v4; pkill -f run_cl_after_d; sleep 1; pkill -f stage_d.py; sleep 4
echo "left: $(pgrep -fc stage_d.py)"
rm -rf ~/af08/runs/stageD ~/af08/D_DONE ~/af08/CL_DONE
mkdir -p ~/fx && cd ~/fx && tar xzf ~/drone-fix-src.tar.gz && rm -rf ~/af08/srcd ~/af08/srccl && cp -r ~/fx/af08/srcd ~/af08/srcd && cp -r ~/fx/af08/srcd ~/af08/srccl
grep -c 'short hold' ~/af08/srcd/stage_a.py ~/af08/srccl/stage_a.py
echo CL_DONE > ~/af08/LAST_MARKER; echo "restart with fixed pilot $(date)" >> ~/af08/runs/setup.log
(setsid nohup bash ~/af08/run_d_after_v4.sh > /dev/null 2>&1 < /dev/null &)
(setsid nohup bash ~/af08/run_cl_after_d.sh > /dev/null 2>&1 < /dev/null &)
sleep 20; echo "stage_d procs: $(pgrep -fc stage_d.py), cl waiter: $(pgrep -fc run_cl_after_d)"
