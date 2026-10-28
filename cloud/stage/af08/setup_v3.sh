#!/usr/bin/env bash
# CPU VM (Ubuntu 24.04): Python venv + simulator deps, then stage A v3 (action-conditioned somatic markers) on 30 workers.
set -u; L=$HOME/af08/runs/setup.log; mkdir -p $HOME/af08/runs
sudo apt-get update -qq >> $L 2>&1; sudo apt-get install -y -qq python3-venv python3-pip libgl1 libegl1 libgomp1 libx11-6 libxext6 >> $L 2>&1
python3 -m venv $HOME/venv >> $L 2>&1; . $HOME/venv/bin/activate
pip install -q --upgrade pip >> $L 2>&1; pip install -q mujoco==3.17.0 numpy scipy shapely mapbox_earcut open3d imageio pillow >> $L 2>&1
python -c "import mujoco, open3d, shapely, mapbox_earcut, scipy; print('deps ok', mujoco.__version__, open3d.__version__)" >> $L 2>&1 || { echo "deps FAILED" >> $L; touch $HOME/af08/V3_DONE; exit 1; }
cd $HOME/af08/src; export OSM_DIR=$HOME/af08/osm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 STAGE_A_ACTIONS=1 STAGE_A_ACT_EVERY=2
echo V3_DONE > $HOME/af08/LAST_MARKER; echo "v3 start $(date)" >> $L
python stage_a.py ${1:-1200} ${2:-200} ${3:-200} ${4:-30} stageA3 > $HOME/af08/runs/v3.log 2>&1; echo "v3 rc=$? $(date)" >> $L
touch $HOME/af08/V3_DONE
