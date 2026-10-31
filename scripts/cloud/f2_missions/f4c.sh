#!/usr/bin/env bash
# F4c (post-hoc, plan F–G 6.4): after F4b/F5 — the system with the neighbourhood scan before touching:
# 'operator' (a person checked the parse and answers the drone's questions) on all 500 missions, then 'full' on the first 300.
set -u; A=$HOME/af08; L=$A/runs/setup.log; WK=${1:-24}
export OSM2_DIR=$A/osm2 MUJOCO_GL=egl YOLO_OFFLINE=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1; . $HOME/venv/bin/activate; cd $A/src_c
pack() { cd $A/runs && tar czf f4c.tgz f4c_*.jsonl f4c_*.log 2>/dev/null; cd $A/src_c; echo "packed f4c $1 $(date)" >> $L; }
until [ -f $A/F4B_DONE ]; do sleep 30; done; echo "f4c start $(date)" >> $L
head -300 $A/runs/f4_plan.jsonl > $A/runs/f4_plan300.jsonl
timeout 7200 python eval_missions.py fly $A/models/detector.pt $A/models/attrs_best.pt $A/runs/f4_plan.jsonl $A/runs/f4c_operator.jsonl $WK operator > $A/runs/f4c_operator.log 2>&1; echo "f4c_operator rc=$? $(date)" >> $L; pack operator
timeout 5400 python eval_missions.py fly $A/models/detector.pt $A/models/attrs_best.pt $A/runs/f4_plan300.jsonl $A/runs/f4c_full.jsonl $WK full > $A/runs/f4c_full.log 2>&1; echo "f4c_full rc=$? $(date)" >> $L; pack full
touch $A/F4C_DONE
