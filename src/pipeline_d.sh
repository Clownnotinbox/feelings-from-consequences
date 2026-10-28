#!/usr/bin/env bash
# Stage D experiments (H9-H11): baselines, small learned policies, zero-shot LLM (3B), LoRA per input condition
# (1.5B, 3 seeds) with lesions. Feelings from the v4 GRU core (seed 1).
cd "$(dirname "$0")"; PY="/c/Users/Пользователь/Documents/Codex/2026-10-20/new-chat/work/adaptation-cuda-venv/Scripts/python.exe"
M=/c/Users/Пользователь/Documents/Codex/2026-10-20/new-chat/work/models; D=../runs/cloud-d/stageD; C=../runs/core4_s1.runtime.npz
for sp in test_familiar test_unfamiliar; do
  "$PY" llm_decide.py --data $D --split $sp --core $C --core-model gru --train --model $M/qwen2.5-3b-instruct --out ../runs/LLM_D_zeroshot3b_$sp.json 2>&1 | grep -v -i warn | cut -c1-260
  echo "zero-shot $sp done $(date)"
done
for s in 1 2 3; do "$PY" llm_train_decide.py --data $D --core $C --core-model gru --model $M/qwen2.5-1.5b-instruct --out ../runs/LLM_D_lora15.json --seed $s 2>&1 | grep -v -i warn | grep "seed\|test_" | cut -c1-260; echo "lora seed $s done $(date)"; done
echo PIPELINE_D_DONE
