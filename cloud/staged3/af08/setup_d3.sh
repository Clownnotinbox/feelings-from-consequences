#!/usr/bin/env bash
# CPU VM (Ubuntu 24.04): stage D3 (FEELINGS_DESIGN 12.6). 1) pilot: 120 flights with a true message, the world forked
# with and without the announced event at the message — stop rule: go on only if the no-event-best plan is wrong with
# the event in >= 20 % (override: touch ~/af08/GO or ~/af08/STOP). 2) 600/150/150 flights of stage D3 data, the v4 GRU
# core's feelings at every decision, packed. Marker D3_DONE.
set -u; A=$HOME/af08; L=$A/runs/setup.log; mkdir -p $A/runs; echo "setup start $(date)" >> $L
P="timeout 3000 pip install -q --timeout 30 --retries 10 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq >> $L 2>&1; sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip libgl1 libegl1 libgomp1 libx11-6 libxext6 >> $L 2>&1
python3 -m venv $HOME/venv >> $L 2>&1; . $HOME/venv/bin/activate
$P --upgrade pip >> $L 2>&1; $P mujoco==3.17.0 numpy scipy shapely mapbox_earcut open3d imageio pillow scikit-learn==1.9.1 >> $L 2>&1
python -c "import mujoco, open3d, shapely, mapbox_earcut, scipy, sklearn; print('deps ok', mujoco.__version__)" >> $L 2>&1 || { echo "deps FAILED" >> $L; touch $A/D3_DONE; exit 1; }
W=$(( $(nproc --all) - 1 )); cd $A/src; export OSM_DIR=$A/osm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DRONE_TRACKER=occlusion STAGE_D_MSG=3
echo "pilot start workers=$W $(date)" >> $L
python d3_pilot.py 120 $W $A/runs/pilotD3.json > $A/runs/pilot.log 2>&1; echo "pilot rc=$? $(date)" >> $L
G=$(python -c "import json; s=json.load(open('$A/runs/pilotD3.json'))['summary']; print('GO' if s['all']['ignore_wrong'] >= .20 else 'STOP')" 2>> $L || echo STOP)
[ -f $A/GO ] && G=GO; [ -f $A/STOP ] && G=STOP; echo "pilot decision $G $(date)" >> $L; touch $A/PILOT_DONE
if [ "$G" = GO ]; then
  python stage_d.py 600 150 150 $W stageD3 > $A/runs/d3.log 2>&1; echo "D3 rc=$? $(date)" >> $L
  python export_d_feelings.py $A/runs/stageD3 $A/cl/core4_s1.runtime.npz $A/runs/stageD3_feelings_core4_s1_gru.json >> $L 2>&1; echo "feelings rc=$? $(date)" >> $L
  cd $A/runs && tar czf $A/runs/d3_small.tgz stageD3/*.json stageD3_feelings_core4_s1_gru.json pilotD3.json pilot.log d3.log setup.log; echo "packed $(date)" >> $L
else
  cd $A/runs && tar czf $A/runs/d3_small.tgz pilotD3.json pilot.log setup.log; echo "stopped by the pilot rule, packed $(date)" >> $L
fi
touch $A/D3_DONE
