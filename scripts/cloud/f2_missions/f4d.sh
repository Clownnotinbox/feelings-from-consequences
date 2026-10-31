#!/usr/bin/env bash
# F4d (plan F–G 6.5): the fixed system alone ('full', F3b parse) on the first 300 F4 missions; starts only after
# acts8 AND the go flag (written by hand once acts8 meets the 6.5 condition).
A=$HOME/af08; L=$A/runs/setup.log; export OSM2_DIR=$A/osm2 MUJOCO_GL=egl YOLO_OFFLINE=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1; . $HOME/venv/bin/activate; cd $A/src_d
until [ -f $A/ACTS8_DONE ] && [ -f $A/F4D_GO ]; do sleep 15; done; echo "f4d start $(date)" >> $L
head -300 $A/runs/f4_plan.jsonl > $A/runs/f4_plan300d.jsonl
timeout 3600 python eval_missions.py fly $A/models/detector.pt $A/models/attrs_best.pt $A/runs/f4_plan300d.jsonl $A/runs/f4d_full.jsonl 24 full > $A/runs/f4d_full.log 2>&1; echo "f4d rc=$? $(date)" >> $L
touch $A/F4D_DONE
