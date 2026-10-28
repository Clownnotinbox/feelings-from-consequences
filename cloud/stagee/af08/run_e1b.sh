#!/usr/bin/env bash
# Stage E step 1, rerun of the VLM part after the smoke run (FEELINGS_DESIGN 13.1, amendment of 25.10 16:55): the model
# often reasons before answering, and 48 new tokens cut the answer off; now 512 and the last answer counts.
set -u; A=$HOME/af08; L=$A/runs/setup.log; M=$HOME/models/qwen2.5-vl-7b; . $HOME/venv/bin/activate; cd $A/src
export MUJOCO_GL=egl OMP_NUM_THREADS=4; echo "rerun start $(date)" >> $L
python -u stage_e_vlm.py --data $A/runs/sceneE --model $M --out $A/runs/vlm_smoke2.json --limit 16 --bs 16 > $A/runs/vlm_smoke2.log 2>&1; echo "vlm smoke2 rc=$? $(date)" >> $L
python -u stage_e_vlm.py --data $A/runs/sceneE --model $M --out $A/runs/vlm7b.json --bs 16 > $A/runs/vlm7b.log 2>&1; echo "vlm rc=$? $(date)" >> $L
cd $A/runs && tar czf e1.tgz sceneE/scenes.jsonl sceneE/img/e000* sceneE/img/e001* embed.json vlm*.json *.log 2> /dev/null; echo "packed $(date)" >> $L; touch $A/E1_DONE
