#!/usr/bin/env bash
# G5 (docs/REAL_DRONE.md sec. 9): after the PX4 SITL build, 3 PX4 instances (SIH quadrotor, headless) and the unchanged brain
# flying each over MAVLink (real/sitl_cosim.py), 4 missions per instance; then the same 12 missions with our own simulated autopilot.
set -u; A=$HOME/af08; L=$A/runs/setup.log; X=$HOME/px4; export OSM2_DIR=$A/osm2 MUJOCO_GL=egl YOLO_OFFLINE=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1
until [ -f $A/PX4_BUILT ]; do sleep 20; done
[ -x $X/build/px4_sitl_default/bin/px4 ] || { echo "px4 run: no binary FAILED" >> $L; exit 1; }
echo "px4 run start $(date)" >> $L; . $HOME/venv/bin/activate
for i in 0 1 2; do
  mkdir -p $A/runs/px4_i$i; (cd $X/build/px4_sitl_default && PX4_SIM_MODEL=sihsim_quadx HEADLESS=1 setsid nohup ./bin/px4 -i $i -d > $A/runs/px4_i$i/px4.log 2>&1 < /dev/null &)
done
sleep 25; for i in 0 1 2; do tail -3 $A/runs/px4_i$i/px4.log >> $L; done
cd $A/real
for i in 0 1 2; do (timeout 3000 python sitl_cosim.py $A/runs/g5_px4.jsonl $((i * 4)) $((i * 4 + 4)) $i > $A/runs/g5_px4_i$i.log 2>&1; echo "g5 px4 i$i rc=$? $(date)" >> $L) & done
wait
timeout 1800 python sitl_cosim.py $A/runs/g5_own.jsonl 0 12 0 own > $A/runs/g5_own.log 2>&1; echo "g5 own rc=$? $(date)" >> $L
pkill -f "[b]in/px4 -i" ; touch $A/G5_DONE
