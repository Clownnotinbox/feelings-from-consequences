#!/usr/bin/env bash
# CPU VM (Ubuntu 24.04): closed loop of the consequence-model markers, seeds 2 and 3 (replication of the seed-1
# result), the same 120 missions per split as before. Marker M_DONE.
set -u; A=$HOME/af08; L=$A/runs/setup.log; mkdir -p $A/runs; echo "setup start $(date)" >> $L
sudo apt-get update -qq >> $L 2>&1; sudo apt-get install -y -qq python3-venv python3-pip libgl1 libegl1 libgomp1 libx11-6 libxext6 >> $L 2>&1
python3 -m venv $HOME/venv >> $L 2>&1; . $HOME/venv/bin/activate
pip install -q --upgrade pip >> $L 2>&1; pip install -q mujoco==3.17.0 numpy scipy shapely mapbox_earcut open3d imageio pillow scikit-learn==1.9.1 >> $L 2>&1
python -c "import mujoco, open3d, shapely, mapbox_earcut, scipy, sklearn; print('deps ok', mujoco.__version__)" >> $L 2>&1 || { echo "deps FAILED" >> $L; touch $A/M_DONE; exit 1; }
cd $A/src; export OSM_DIR=$A/osm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DRONE_TRACKER=occlusion
echo M_DONE > $A/LAST_MARKER; W=$(( $(nproc) - 1 ))
for s in 2 3; do
  python closed_loop.py --markers-fwd $A/cl/markers4_s$s.fwd.pkl --variants markers_fwd@fa5 --n 120 --offset 0 --workers $W --out $A/runs/CLOSED_LOOP_v4_m$s.json > $A/runs/cl_m$s.log 2>&1
  echo "m$s rc=$? $(date)" >> $L
done
touch $A/M_DONE
