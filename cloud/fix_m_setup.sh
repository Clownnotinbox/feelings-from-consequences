#!/usr/bin/env bash
# Run as a FILE on the GPU VM: stop the stalled setup (by PID), add pip timeouts, restart the setup.
for p in $(pgrep -f "af08/setup_m.sh"); do [ "$p" != "$$" ] && kill $p; done
for p in $(pgrep -f "pip install"); do kill $p; done; sleep 2
sed -i 's/pip install -q /timeout 1200 pip install -q --timeout 30 --retries 10 /g' ~/af08/setup_m.sh
grep -c "timeout 1200" ~/af08/setup_m.sh
rm -f ~/af08/M_DONE; echo "restart setup $(date)" >> ~/af08/runs/setup.log
(setsid nohup bash ~/af08/setup_m.sh > /dev/null 2>&1 < /dev/null &); sleep 3; pgrep -fc "af08/setup_m.sh"
