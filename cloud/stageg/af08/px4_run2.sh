#!/usr/bin/env bash
# G5b (28.10, after G5a): the same 12 missions on PX4 SITL with the brain sending POSITION targets in precise phases (landing, hook,
# drop) and touchdowns judged also by height over the surface; the brain from src_px4 (src + the 'dxy'/'precise' command keys).
set -u; A=$HOME/af08; L=$A/runs/setup.log; X=$HOME/px4; export OSM2_DIR=$A/osm2 MUJOCO_GL=egl YOLO_OFFLINE=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 BRAIN_SRC=$A/src_px4
echo "px4 run2 start $(date)" >> $L; . $HOME/venv/bin/activate
for i in 0 1 2; do (cd $X/build/px4_sitl_default && PX4_SIM_MODEL=sihsim_quadx HEADLESS=1 setsid nohup ./bin/px4 -i $i -d > $A/runs/px4_i$i/px4b.log 2>&1 < /dev/null &); done
sleep 25; cd $A/real
for i in 0 1 2; do (timeout 3000 nice -n -0 python sitl_cosim.py $A/runs/g5b_px4.jsonl $((i * 4)) $((i * 4 + 4)) $i > $A/runs/g5b_px4_i$i.log 2>&1; echo "g5b px4 i$i rc=$? $(date)" >> $L) & done
wait; pkill -f "[b]in/px4 -i"; touch $A/G5B_DONE
