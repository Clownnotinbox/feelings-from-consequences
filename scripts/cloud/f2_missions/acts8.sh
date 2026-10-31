#!/usr/bin/env bash
# After F4c 'operator' (27.10, evening): every action on the TRAINING / validation places with the fixes of 19:50-20:10
# (horizontal velocity integrator in the autopilot, hangar-floor landing scored as 'enter', hangar checked from the side,
# no candidate beyond 170 m of the rough point); replaces the F4c 'full' run (plan F–G 6.4, note of 20:15).
A=$HOME/af08; L=$A/runs/setup.log; export OSM2_DIR=$A/osm2 MUJOCO_GL=egl YOLO_OFFLINE=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1; . $HOME/venv/bin/activate; cd $A/src_d
until [ -f $A/F4C_DONE ]; do sleep 20; done; echo "acts8 start $(date)" >> $L
timeout 3000 python test_actions.py $A/runs/acts8.jsonl 20 24 > $A/runs/acts8.log 2>&1; echo "acts8 rc=$? $(date)" >> $L; touch $A/ACTS8_DONE
