#!/usr/bin/env bash
# GPU VM (Ubuntu 24.04 CUDA, one RTX 4090): stage D2 (H12) LoRA — conditions FT, FTM, K, KM, 3 seeds (3 parallel
# processes), 6 epochs; feelings from the cache, decisions with messages. Then Qwen2.5-7B zero-shot (LoRA with 0 epochs =
# the base model) on K, KM, FTM: the LLM's own knowledge of the world without any tuning. Marker LLM2_DONE.
set -u; A=$HOME/af08; L=$A/runs/setup.log; mkdir -p $A/runs; echo "setup start $(date)" >> $L
nvidia-smi -L >> $L 2>&1 || { echo "no GPU" >> $L; echo LLM2_DONE > $A/LAST_MARKER; touch $A/LLM2_DONE; exit 1; }
P="timeout 1500 pip install -q --timeout 30 --retries 10"; MIR="-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq >> $L 2>&1; sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip python3-dev build-essential >> $L 2>&1
python3 -m venv $HOME/venv >> $L 2>&1; . $HOME/venv/bin/activate; $P $MIR --upgrade pip >> $L 2>&1
$P torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126 >> $L 2>&1
$P $MIR transformers==4.57.6 peft==0.18.1 accelerate scikit-learn scipy numpy huggingface_hub >> $L 2>&1
python -c "import torch,transformers,peft,sklearn;print('deps ok',torch.__version__,torch.cuda.device_count(),transformers.__version__,peft.__version__)" >> $L 2>&1 || { echo "deps FAILED" >> $L; echo LLM2_DONE > $A/LAST_MARKER; touch $A/LLM2_DONE; exit 1; }
M=$HOME/models/qwen2.5-1.5b-instruct; G="from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-1.5B-Instruct', local_dir='$M')"
python -c "$G" >> $L 2>&1 || HF_ENDPOINT=https://hf-mirror.com python -c "$G" >> $L 2>&1
ls $M/*.safetensors >> $L 2>&1 || { echo "model FAILED" >> $L; echo LLM2_DONE > $A/LAST_MARKER; touch $A/LLM2_DONE; exit 1; }
echo "deps+model ready $(date)" >> $L; echo LLM2_DONE > $A/LAST_MARKER; cd $A/src
for s in 1 2 3; do
  (python -u llm_train_decide2.py --data $A/stageD2 --core core4_s1.runtime.npz --core-model gru --feelings $A/stageD2_feelings_core4_s1_gru.json      --model $M --out $A/runs/LLM_D2M_s$s.json --seed $s --conds K,KM,FT,FTM --epochs 6 --ckpt 1 > $A/runs/llm_s$s.log 2>&1; echo "seed $s rc=$? $(date)" >> $L) &
  sleep 30
done
M7=$HOME/models/qwen2.5-7b-instruct; G7="from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-7B-Instruct', local_dir='$M7')"
(python -c "$G7" >> $L 2>&1 || HF_ENDPOINT=https://hf-mirror.com python -c "$G7" >> $L 2>&1; echo "7B download rc=$? $(date)" >> $L) &
wait; echo "training done $(date)" >> $L
python -u llm_train_decide2.py --data $A/stageD2 --core core4_s1.runtime.npz --core-model gru --feelings $A/stageD2_feelings_core4_s1_gru.json   --model $M7 --out $A/runs/LLM_D2M_zs7b.json --seed 1 --conds K,KM,FTM --epochs 0 --ckpt 0 > $A/runs/llm_zs7b.log 2>&1; echo "7B zero-shot rc=$? $(date)" >> $L
touch $A/LLM2_DONE
