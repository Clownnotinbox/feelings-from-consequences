#!/usr/bin/env bash
# CPU VM (Ubuntu 24.04): stage D2 (H12) — slow decisions with words about the future world (messages.py), 600/150/150
# flights of 120 s, 5 options each rolled out to the end. Then the v4 GRU core's feelings at every decision (so the
# LLM can be trained elsewhere from small files). Marker D2_DONE.
set -u; A=$HOME/af08; L=$A/runs/setup.log; mkdir -p $A/runs; echo "setup start $(date)" >> $L
P="timeout 1200 pip install -q --timeout 30 --retries 10 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq >> $L 2>&1; sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip libgl1 libegl1 libgomp1 libx11-6 libxext6 >> $L 2>&1
python3 -m venv $HOME/venv >> $L 2>&1; . $HOME/venv/bin/activate
$P --upgrade pip >> $L 2>&1; $P mujoco==3.17.0 numpy scipy shapely mapbox_earcut open3d imageio pillow scikit-learn==1.9.1 >> $L 2>&1
python -c "import mujoco, open3d, shapely, mapbox_earcut, scipy, sklearn; print('deps ok', mujoco.__version__)" >> $L 2>&1 || { echo "deps FAILED" >> $L; echo D2_DONE > $A/LAST_MARKER; touch $A/D2_DONE; exit 1; }
W=$(( $(nproc --all) - 1 )); cd $A/src; export OSM_DIR=$A/osm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DRONE_TRACKER=occlusion STAGE_D_MSG=1
echo D2_DONE > $A/LAST_MARKER; echo "D2 start workers=$W $(date)" >> $L
python stage_d.py 600 150 150 $W stageD2 > $A/runs/d2.log 2>&1; echo "D2 rc=$? $(date)" >> $L
python export_d_feelings.py $A/runs/stageD2 $A/cl/core4_s1.runtime.npz $A/runs/stageD2_feelings_core4_s1_gru.json >> $L 2>&1; echo "feelings rc=$? $(date)" >> $L
cd $A/runs && tar czf $A/runs/d2_small.tgz stageD2/*.json stageD2_feelings_core4_s1_gru.json d2.log setup.log && tar cf $A/runs/d2_npz.tar stageD2/*.npz; echo "packed $(date)" >> $L
touch $A/D2_DONE
