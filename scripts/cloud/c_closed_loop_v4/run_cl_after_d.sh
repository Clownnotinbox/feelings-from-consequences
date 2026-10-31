#!/usr/bin/env bash
# On the drone VM: after stage D, closed-loop flights of system v4 (occlusion tracker, v4 cores and markers).
set -u; L=$HOME/af08/runs/setup.log
while [ ! -f $HOME/af08/D_DONE ]; do sleep 60; done
. $HOME/venv/bin/activate; cd $HOME/af08/srccl; export OSM_DIR=$HOME/af08/osm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DRONE_TRACKER=occlusion
echo "CL start $(date)" >> $L; R=$HOME/af08/cl
python closed_loop.py --core $R/core4_s1.runtime.npz --core-model gru --markers $R/markers4_s1.runtime.npz --markers-fwd $R/markers4_s1.fwd.pkl \
  --n ${1:-60} --workers ${2:-30} --out $HOME/af08/runs/CLOSED_LOOP_v4.json --variants v4 > $HOME/af08/runs/cl.log 2>&1; echo "CL rc=$? $(date)" >> $L
touch $HOME/af08/CL_DONE
