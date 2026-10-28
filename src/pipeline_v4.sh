#!/usr/bin/env bash
# Data v4 (cloud stageA4 + laptop stageA4L, occlusion tracker): cores (3 seeds, GRU without class weights), fair
# analysis by contact kind, observability on the test episodes, action markers (3 seeds, cloud data has them).
cd "$(dirname "$0")"; PY="/c/Users/Пользователь/Documents/Codex/2026-10-20/new-chat/work/adaptation-cuda-venv/Scripts/python.exe"
D=../runs/cloud-v4/stageA4,../runs/stageA4L
for s in 1 2 3; do "$PY" train_core.py --data $D --out ../runs/core4_s$s.json --seed $s --no-fly --no-posweight --epochs 40 2>&1 | grep -v Warning | grep -v "gru epoch" | cut -c1-160 | tail -2; echo "core4 seed $s $(date)"; done
"$PY" analyze_core.py cloud-v4/stageA4,stageA4L core4_s 2>&1 | grep -v Warn
OBS_SPLIT=test_ "$PY" analyze_observability.py $D ../runs/core4_s1.runtime.npz 2>&1 | grep -v Warn | tail -1 | cut -c1-900
for s in 1 2 3; do "$PY" train_markers.py ../runs/cloud-v4/stageA4 ../runs/markers4_s$s.json $s 2>&1 | grep -v "Warn\|epoch" | cut -c1-200 | tail -4; echo "markers4 seed $s $(date)"; done
echo PIPELINE_V4_DONE
