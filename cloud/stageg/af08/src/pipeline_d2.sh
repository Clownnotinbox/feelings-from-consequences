#!/usr/bin/env bash
# Stage D v2 (H9-H11): LoRA 1.5B with a validation split, shuffled option order, feelings in words or as soft slots
# (and a telemetry-slot control), 3 seeds; lesions. Feelings from the v4 GRU core (seed 1). 8 epochs, no early stop.
cd "$(dirname "$0")"; PY="/c/Users/Пользователь/Documents/Codex/2026-10-20/new-chat/work/adaptation-cuda-venv/Scripts/python.exe"
M=/c/Users/Пользователь/Documents/Codex/2026-10-20/new-chat/work/models; D=../runs/cloud-d/stageD; C=../runs/core4_s1.runtime.npz
for s in 1 2 3; do
  "$PY" -u llm_train_decide2.py --data $D --core $C --core-model gru --model $M/qwen2.5-1.5b-instruct --out ../runs/LLM_D2_lora15.json --seed $s --conds F,Fs,Ts,FT,FsT,N,T 2>&1 | grep --line-buffered -v -i warn | grep --line-buffered "seed\|test_\|Error\|error"
  echo "v2 seed $s done $(date)"
done
echo PIPELINE_D2_DONE
