#!/usr/bin/env bash
# Drone v3 VM: wait for SSH, upload the bundle, run setup_v3.sh detached on the VM (it writes LAST_MARKER=V3_DONE).
export MSYS_NO_PATHCONV=1; H=$1; cd "$(dirname "$0")"
until python ssh.py $H run "true" 20 > /dev/null 2>&1; do sleep 20; done; echo "ssh up $(date)"
python ssh.py $H put drone-v3-bundle.tar.gz /home/ubuntu/drone-v3-bundle.tar.gz && echo "uploaded $(date)"
python ssh.py $H run "cd ~ && tar xzf drone-v3-bundle.tar.gz && sed -i 's/\r$//' ~/af08/setup_v3.sh && (setsid nohup bash ~/af08/setup_v3.sh > /dev/null 2>&1 < /dev/null &); echo started" 60
echo "launched $(date)"
