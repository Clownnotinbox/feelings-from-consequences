#!/usr/bin/env bash
# GPU VM (Ubuntu 24.04 CUDA): venv + torch/transformers/peft, Qwen2.5-1.5B-Instruct, then stage D v2 LoRA — the 3 seeds
# as parallel processes (separate result files), feelings from the cache made on the laptop. Marker LLM_DONE.
set -u; A=$HOME/af08; L=$A/runs/setup.log; mkdir -p $A/runs; echo "setup start $(date)" >> $L
nvidia-smi >> $L 2>&1 || { echo "no GPU" >> $L; touch $A/LLM_DONE; exit 1; }
sudo apt-get update -qq >> $L 2>&1; sudo apt-get install -y -qq python3-venv python3-pip >> $L 2>&1
python3 -m venv $HOME/venv >> $L 2>&1; . $HOME/venv/bin/activate; pip install -q --upgrade pip >> $L 2>&1
pip install -q torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126 >> $L 2>&1 || pip install -q torch --index-url https://download.pytorch.org/whl/cu126 >> $L 2>&1
pip install -q transformers==4.57.6 peft==0.18.1 accelerate scikit-learn scipy numpy huggingface_hub >> $L 2>&1
python -c "import torch,transformers,peft;print('deps ok',torch.__version__,torch.cuda.is_available(),transformers.__version__,peft.__version__)" >> $L 2>&1 || { echo "deps FAILED" >> $L; touch $A/LLM_DONE; exit 1; }
M=$HOME/models/qwen2.5-1.5b-instruct; G="from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-1.5B-Instruct', local_dir='$M')"
python -c "$G" >> $L 2>&1 || HF_ENDPOINT=https://hf-mirror.com python -c "$G" >> $L 2>&1
ls $M/*.safetensors >> $L 2>&1 || { echo "model FAILED" >> $L; touch $A/LLM_DONE; exit 1; }
echo "deps+model ready $(date)" >> $L; echo LLM_DONE > $A/LAST_MARKER; cd $A/src; cp $A/runs/LLM_D2_lora15.json $A/runs/LLM_D2_s1.json
for s in 1 2 3; do
  (python -u llm_train_decide2.py --data $A/stageD --core core4_s1.runtime.npz --core-model gru --feelings $A/stageD_feelings_core4_s1_gru.json --model $M \
     --out $A/runs/LLM_D2_s$s.json --seed $s --conds F,Fs,Ts,FT,FsT,N,T --ckpt 1 > $A/runs/llm_s$s.log 2>&1; echo "seed $s rc=$? $(date)" >> $L) &
  sleep 60
done
wait; echo "all seeds done $(date)" >> $L; touch $A/LLM_DONE
