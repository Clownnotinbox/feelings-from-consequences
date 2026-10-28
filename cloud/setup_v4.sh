#!/usr/bin/env bash
# CPU VM (Ubuntu 24.04): Python venv + simulator deps, then stage A v4 (occlusion tracker, action-conditioned somatic markers) on 30 workers.
set -u; L=$HOME/af08/runs/setup.log; mkdir -p $HOME/af08/runs
. $HOME/venv/bin/activate; python -c "import mujoco, open3d, shapely, mapbox_earcut, scipy; print('deps ok', mujoco.__version__, open3d.__version__)" >> $L 2>&1 || { echo "deps FAILED" >> $L; touch $HOME/af08/V4_DONE; exit 1; }
cd $HOME/af08/src; export OSM_DIR=$HOME/af08/osm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 STAGE_A_ACTIONS=1 STAGE_A_ACT_EVERY=2 DRONE_TRACKER=occlusion
echo V4_DONE > $HOME/af08/LAST_MARKER; echo "v4 start $(date)" >> $L
python stage_a.py ${1:-1200} ${2:-200} ${3:-200} ${4:-30} stageA4 > $HOME/af08/runs/v4.log 2>&1; echo "v4 rc=$? $(date)" >> $L
touch $HOME/af08/V4_DONE
