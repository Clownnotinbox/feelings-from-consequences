#!/usr/bin/env bash
# F4/F5 (MISSIONS_DESIGN_RU.md 6, 6.1): 500 missions on the test places (tokyo_res, kazan_novosav), actions balanced, texts parsed by
# the trained parser (ADAPTER), trained perception; variants in the recorded order: full -> reflex (no core) -> telemetry (no feelings).
# usage: f4.sh ADAPTER [N] [WORKERS]
set -u; A=$HOME/af08; L=$A/runs/setup.log; ADAPTER=$1; N=${2:-500}; WK=${3:-20}; M1=$HOME/models/qwen1.5b
export OSM2_DIR=$A/osm2 MUJOCO_GL=egl YOLO_OFFLINE=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1; . $HOME/venv/bin/activate; cd $A/src
pack() { cd $A/runs && tar czf f4.tgz f4_plan.jsonl f4_*.jsonl f4_*.log 2>/dev/null; cd $A/src; echo "packed f4 $1 $(date)" >> $L; }
echo "f4 start $(date) adapter=$ADAPTER N=$N workers=$WK" >> $L
timeout 2400 python eval_missions.py parse $ADAPTER $M1 $A/runs/f4_plan.jsonl $N > $A/runs/f4_plan.log 2>&1; echo "f4_plan rc=$? $(date)" >> $L; pack plan
for V in full reflex telemetry; do
  timeout 14400 python eval_missions.py fly $A/models/detector.pt $A/models/attrs_best.pt $A/runs/f4_plan.jsonl $A/runs/f4_$V.jsonl $WK $V > $A/runs/f4_$V.log 2>&1; echo "f4_$V rc=$? $(date)" >> $L; pack $V
done
touch $A/F4_DONE
