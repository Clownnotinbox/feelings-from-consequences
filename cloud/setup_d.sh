#!/usr/bin/env bash
# CPU VM (Ubuntu 24.04): venv + simulator deps, then stage D (slow decisions: 5 plan options rolled out to the end).
set -u; L=$HOME/af08/runs/setup.log; mkdir -p $HOME/af08/runs
sudo apt-get update -qq >> $L 2>&1; sudo apt-get install -y -qq python3-venv python3-pip libgl1 libegl1 libgomp1 libx11-6 libxext6 >> $L 2>&1
python3 -m venv $HOME/venv >> $L 2>&1; . $HOME/venv/bin/activate
pip install -q --upgrade pip >> $L 2>&1; pip install -q mujoco==3.17.0 numpy scipy shapely mapbox_earcut open3d imageio pillow >> $L 2>&1
python -c "import mujoco, open3d, shapely, mapbox_earcut, scipy; print('deps ok', mujoco.__version__, open3d.__version__)" >> $L 2>&1 || { echo "deps FAILED" >> $L; touch $HOME/af08/D_DONE; exit 1; }
cd $HOME/af08/src; export OSM_DIR=$HOME/af08/osm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DRONE_TRACKER=occlusion
echo D_DONE > $HOME/af08/LAST_MARKER; echo "D start $(date)" >> $L
python stage_d.py ${1:-400} ${2:-100} ${3:-100} ${4:-30} stageD > $HOME/af08/runs/d.log 2>&1; echo "D rc=$? $(date)" >> $L
touch $HOME/af08/D_DONE
