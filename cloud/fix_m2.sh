#!/usr/bin/env bash
# Run as a FILE on the CPU VM: install from the aliyun PyPI mirror (PyPI itself trickles at ~1 MB/s here), restart setup.
for p in $(pgrep -f "af08/setup_m.sh"); do kill $p; done; for p in $(pgrep -f "pip install"); do kill $p; done; sleep 2
sed -i 's#--timeout 30 --retries 10 #--timeout 30 --retries 10 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com #g' ~/af08/setup_m.sh
grep -c aliyun ~/af08/setup_m.sh; rm -f ~/af08/M_DONE; echo "restart via mirror $(date)" >> ~/af08/runs/setup.log
(setsid nohup bash ~/af08/setup_m.sh > /dev/null 2>&1 < /dev/null &); sleep 2; echo restarted
