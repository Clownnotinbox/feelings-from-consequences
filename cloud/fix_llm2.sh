#!/usr/bin/env bash
# Run as a FILE on the GPU VM: stop training, install Python headers (triton compiles a helper at first CUDA use), restart.
for p in $(pgrep -f "af08/run_llm.sh"); do kill $p; done
for p in $(pgrep -f "llm_train_decide2"); do kill $p; done; sleep 3
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-dev build-essential >> ~/af08/runs/setup.log 2>&1
ls /usr/include/python3.12/Python.h || { echo "no Python.h"; exit 1; }
. ~/venv/bin/activate; python -c "import torch; x=torch.randn(64,64,device='cuda'); print('cuda ok', float((x@x).sum())>-1e9)"
echo "python3-dev installed, restart $(date)" >> ~/af08/runs/setup.log
(setsid nohup bash ~/af08/run_llm.sh > /dev/null 2>&1 < /dev/null &); sleep 2; echo restarted
