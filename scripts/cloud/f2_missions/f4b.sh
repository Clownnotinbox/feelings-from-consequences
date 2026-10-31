#!/usr/bin/env bash
# F4b/F5 (plan F–G 6.3): the same 500 missions and parses (runs/f4_plan.jsonl), the fixed system (src_b), variants full -> reflex -> telemetry.
set -u; A=$HOME/af08; L=$A/runs/setup.log; WK=${1:-20}
export OSM2_DIR=$A/osm2 MUJOCO_GL=egl YOLO_OFFLINE=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1; . $HOME/venv/bin/activate; cd $A/src_b
pack() { cd $A/runs && tar czf f4b.tgz f4_plan.jsonl f4_full.jsonl f4b_*.jsonl f4b_*.log 2>/dev/null; cd $A/src_b; echo "packed f4b $1 $(date)" >> $L; }
while pgrep -f 'eval_missions.py fly .*f4_full' > /dev/null; do sleep 30; done; echo "f4b start $(date)" >> $L
for V in full reflex telemetry; do
  timeout 14400 python eval_missions.py fly $A/models/detector.pt $A/models/attrs_best.pt $A/runs/f4_plan.jsonl $A/runs/f4b_$V.jsonl $WK $V > $A/runs/f4b_$V.log 2>&1; echo "f4b_$V rc=$? $(date)" >> $L; pack $V
done
touch $A/F4B_DONE
