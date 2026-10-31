#!/usr/bin/env bash
# GPU VM (Ubuntu 24.04, one RTX 4090): stage D3 (FEELINGS_DESIGN 12.6). 1) Qwen-1.5B + LoRA decider KM (core's judgement +
# message), 3 seeds in parallel, 6 epochs; tests: as is, the never-trained wording, message removed, judgement shuffled.
# 2) Qwen-7B without training as the reader of every message (both wordings) -> numbers for the core's policy.
# 3) Qwen-7B without training as the decider KM. Marker D3LLM_DONE; results packed in runs/r3.tgz.
set -u; A=$HOME/af08; L=$A/runs/setup.log; mkdir -p $A/runs; echo "setup start $(date)" >> $L
nvidia-smi -L >> $L 2>&1 || { echo "no GPU" >> $L; touch $A/D3LLM_DONE; exit 1; }
P="timeout 1500 pip install -q --timeout 30 --retries 10"; MIR="-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq >> $L 2>&1; sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip python3-dev build-essential >> $L 2>&1
python3 -m venv $HOME/venv >> $L 2>&1; . $HOME/venv/bin/activate; $P $MIR --upgrade pip >> $L 2>&1
$P torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126 >> $L 2>&1
$P $MIR transformers==4.57.6 peft==0.18.1 accelerate scikit-learn==1.9.1 scipy numpy huggingface_hub >> $L 2>&1
python -c "import torch,transformers,peft,sklearn;print('deps ok',torch.__version__,torch.cuda.device_count(),transformers.__version__,peft.__version__)" >> $L 2>&1 || { echo "deps FAILED" >> $L; touch $A/D3LLM_DONE; exit 1; }
M=$HOME/models/qwen2.5-1.5b-instruct; G="from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-1.5B-Instruct', local_dir='$M')"
python -c "$G" >> $L 2>&1 || HF_ENDPOINT=https://hf-mirror.com python -c "$G" >> $L 2>&1
ls $M/*.safetensors >> $L 2>&1 || { echo "model FAILED" >> $L; touch $A/D3LLM_DONE; exit 1; }
echo "deps+model ready $(date)" >> $L; cd $A/src; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
C="--data $A/stageD3 --core core4_s1.runtime.npz --core-model gru --feelings $A/stageD3_feelings_core4_s1_gru.json"
python -u llm_train_decide2.py $C --model $M --out $A/runs/smoke.json --seed 9 --conds KM --epochs 1 --limit 80 > $A/runs/smoke.log 2>&1; echo "smoke rc=$? $(date)" >> $L
for s in 1 2 3; do
  (python -u llm_train_decide2.py $C --model $M --out $A/runs/LLM_D3_s$s.json --seed $s --conds KM --epochs 6 --ckpt 1 > $A/runs/llm_s$s.log 2>&1; echo "seed $s rc=$? $(date)" >> $L) &
  sleep 20
done
M7=$HOME/models/qwen2.5-7b-instruct; G7="from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-7B-Instruct', local_dir='$M7')"
(python -c "$G7" >> $L 2>&1 || HF_ENDPOINT=https://hf-mirror.com python -c "$G7" >> $L 2>&1; echo "7B download rc=$? $(date)" >> $L) &
wait; echo "training done $(date)" >> $L
python -u llm_read_messages.py --data $A/stageD3 --model $M7 --out $A/runs/read7b_smoke.json --limit 8 > $A/runs/read7b.log 2>&1; echo "reader smoke rc=$? $(date)" >> $L
python -u llm_read_messages.py --data $A/stageD3 --model $M7 --out $A/runs/read7b.json >> $A/runs/read7b.log 2>&1; echo "reader rc=$? $(date)" >> $L
python -u llm_train_decide2.py $C --model $M7 --out $A/runs/LLM_D3_zs7b.json --seed 1 --conds KM --epochs 0 --ckpt 0 > $A/runs/llm_zs7b.log 2>&1; echo "7B zero-shot rc=$? $(date)" >> $L
cd $A/runs && tar czf r3.tgz LLM_D3_*.json read7b*.json *.log; echo "packed $(date)" >> $L
touch $A/D3LLM_DONE
