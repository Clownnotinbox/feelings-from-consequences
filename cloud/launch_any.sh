#!/usr/bin/env bash
# usage: launch_any.sh HOST BUNDLE.tar.gz SETUP_SCRIPT_NAME — wait for SSH, upload the bundle, run af08/SETUP detached.
export MSYS_NO_PATHCONV=1; H=$1; B=$2; S=$3; cd "$(dirname "$0")"
ssh-keygen -R $H -f known_hosts_af08 > /dev/null 2>&1  # a fresh VM can reuse an IP of an old one: drop its stale host key
k=0; until python ssh.py $H run "true" 20 > /tmp/ssh_try.log 2>&1; do k=$((k+1)); [ $((k % 6)) -eq 0 ] && echo "no ssh yet ($k): $(tail -1 /tmp/ssh_try.log)"; sleep 20; done; echo "ssh up $H $(date)"
python ssh.py $H put $B /home/ubuntu/$B && echo "uploaded $(date)"
python ssh.py $H run "cd ~ && tar xzf $B && sed -i 's/\r$//' ~/af08/$S && (setsid nohup bash ~/af08/$S > /dev/null 2>&1 < /dev/null &); sleep 2; echo started; ls ~/af08" 60
echo "launched $H $(date)"
