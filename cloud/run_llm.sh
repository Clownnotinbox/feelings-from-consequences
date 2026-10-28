#!/usr/bin/env bash
# GPU VM: (re)start stage D v2 training only — deps and model are in place. 3 seeds in parallel, marker LLM_DONE.
set -u; A=$HOME/af08; L=$A/runs/setup.log; M=$HOME/models/qwen2.5-1.5b-instruct; . $HOME/venv/bin/activate
rm -f $A/LLM_DONE; echo LLM_DONE > $A/LAST_MARKER; echo "training start $(date)" >> $L; cd $A/src
[ -f $A/runs/LLM_D2_s1.json ] || cp $A/runs/LLM_D2_lora15.json $A/runs/LLM_D2_s1.json
for s in 1 2 3; do
  (python -u llm_train_decide2.py --data $A/stageD --core core4_s1.runtime.npz --core-model gru --feelings $A/stageD_feelings_core4_s1_gru.json --model $M \
     --out $A/runs/LLM_D2_s$s.json --seed $s --conds F,Fs,Ts,FT,FsT,N,T --ckpt 1 > $A/runs/llm_s$s.log 2>&1; echo "seed $s rc=$? $(date)" >> $L) &
  sleep 45
done
wait; echo "all seeds done $(date)" >> $L; touch $A/LLM_DONE
