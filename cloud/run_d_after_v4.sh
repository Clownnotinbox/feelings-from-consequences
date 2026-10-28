#!/usr/bin/env bash
# On the drone VM: wait for stage A v4 to finish, then run stage D from its own source copy (~/af08/srcd).
set -u; L=$HOME/af08/runs/setup.log
while [ ! -f $HOME/af08/V4_DONE ]; do sleep 60; done
. $HOME/venv/bin/activate; cd $HOME/af08/srcd; export OSM_DIR=$HOME/af08/osm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DRONE_TRACKER=occlusion
echo "D start $(date)" >> $L
python stage_d.py ${1:-400} ${2:-100} ${3:-100} ${4:-30} stageD > $HOME/af08/runs/d.log 2>&1; echo "D rc=$? $(date)" >> $L
touch $HOME/af08/D_DONE
