#!/usr/bin/env bash
# The second action batch (all fixes of 27.10), after the 7B paraphrases free the GPU; 12 workers next to the parser training.
A=$HOME/af08; L=$A/runs/setup.log; export OSM2_DIR=$A/osm2 MUJOCO_GL=egl YOLO_OFFLINE=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1; . $HOME/venv/bin/activate; cd $A/src
until grep -q 'f3b_para_val rc=' <(sed -n '/f3b2 start/,$p' $L); do sleep 20; done; sleep 30; echo "acts6 start $(date)" >> $L
timeout 5400 python test_actions.py $A/runs/acts6.jsonl 6 12 > $A/runs/acts6.log 2>&1; echo "acts6 rc=$? $(date)" >> $L; touch $A/ACTS6_DONE
