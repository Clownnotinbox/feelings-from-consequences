#!/usr/bin/env bash
# Stage G (plan F–G 6.6) on one VM (RTX 3090, 32 vCPU): setup -> smoke (every variant, short flights) -> G1 (door
# finder) -> G2 (final test: full + reflex, then operator) -> G3 (slow feelings, hard conditions) -> G4 (false / delayed danger).
# PX4 SITL is built in parallel at low priority (px4.sh). Packs after every step; a local waiter pulls g.tgz.
set -u; A=$HOME/af08; L=$A/runs/setup.log; mkdir -p $A/runs; echo "setup start $(date)" >> $L
export OSM2_DIR=$A/osm2 MUJOCO_GL=egl PIP_NO_CACHE_DIR=1 YOLO_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONIOENCODING=utf-8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { echo "$1 rc=$2 $(date)" >> $L; [ "$2" != "0" ] && { echo "$1 FAILED" >> $L; tail -25 $A/runs/$1.log >> $L 2>/dev/null; }; return 0; }
pack() { cd $A/runs && tar czf g.tgz *.log *.json *.jsonl 2>/dev/null; cd $A/src; echo "packed $1 $(date)" >> $L; }
nvidia-smi -L >> $L 2>&1 || { echo "no GPU FAILED" >> $L; touch $A/G_DONE; exit 1; }
P="timeout 3000 pip install -q --timeout 30 --retries 10"; MIR="-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq >> $L 2>&1
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip python3-dev build-essential libegl1 libgl1 libosmesa6 libglx-mesa0 libglib2.0-0 libgomp1 git cmake ninja-build rsync unzip zip file libxml2-utils >> $L 2>&1
(bash $A/px4.sh > $A/runs/px4_build.log 2>&1 &)  # PX4 SITL at low priority on the side (not a hypothesis: the readiness check)
python3 -m venv $HOME/venv >> $L 2>&1; . $HOME/venv/bin/activate; $P $MIR --upgrade pip >> $L 2>&1; $P $MIR huggingface_hub >> $L 2>&1
M1=$HOME/models/qwen1.5b
(timeout 1500 python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-1.5B-Instruct', local_dir='$M1')" >> $A/runs/dl.log 2>&1 || HF_ENDPOINT=https://hf-mirror.com timeout 2400 python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-1.5B-Instruct', local_dir='$M1')" >> $A/runs/dl.log 2>&1; echo "download qwen1.5b rc=$? $(du -sh $M1 | cut -f1) $(date)" >> $L) &
DL=$!
$P torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu126 >> $L 2>&1
timeout 1200 pip install -q --timeout 20 --retries 3 transformers==4.57.6 accelerate peft mujoco==3.17.0 PyOpenGL pillow numpy shapely mapbox_earcut scipy ultralytics open3d scikit-learn==1.9.1 pymavlink >> $L 2>&1 || $P $MIR transformers==4.57.6 accelerate peft mujoco==3.17.0 PyOpenGL pillow numpy shapely mapbox_earcut scipy ultralytics open3d scikit-learn==1.9.1 pymavlink >> $L 2>&1
python -c "import torch,transformers,peft,mujoco,ultralytics,shapely,mapbox_earcut,open3d,sklearn;print('deps ok',torch.__version__,torch.cuda.is_available(),mujoco.__version__,sklearn.__version__)" >> $L 2>&1 || { echo "deps FAILED" >> $L; touch $A/G_DONE; exit 1; }
cd $A/src; NC=$(nproc --all); MEMG=$(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo); WK=$((NC - 6)); [ $((MEMG / 2)) -lt $WK ] && WK=$((MEMG / 2)); export OMP_NUM_THREADS=1; DET=$A/models/detector.pt; ATT=$A/models/attrs_best.pt; echo "cores $NC mem ${MEMG}G workers $WK" >> $L
# ---------------- SMOKE: the door finder on one mission; 2 missions of each plan, every variant, 40 s flights (a real dry run, rule 59)
timeout 900 python test_door.py $A/runs/g1a_smoke.json 1 > $A/runs/smoke_door.log 2>&1; step smoke_door $?
timeout 900 python eval_missions.py parse - - $A/runs/smoke_plan.jsonl 4 H 4 > $A/runs/smoke_plan.log 2>&1; step smoke_plan $?
EVAL_TMAX=40 timeout 1200 python eval_missions.py fly $DET $ATT $A/runs/smoke_plan.jsonl $A/runs/smoke_fly.jsonl $WK full,operator,telemetry,reflex,slow_none,nolost,noenergy,false_alarm,delayed > $A/runs/smoke_fly.log 2>&1; step smoke_fly $?
python - >> $L 2>&1 <<'PY'
import json; R = [json.loads(l) for l in open('/home/ubuntu/af08/runs/smoke_fly.jsonl')]; E = [r for r in R if r.get('status') == 'error']
print('smoke flights', len(R), 'errors', len(E)); [print(r['variant'], r['error'][-600:]) for r in E[:3]]
open('/home/ubuntu/af08/runs/SMOKE_' + ('OK' if R and not E else 'FAILED'), 'w').write('')
PY
pack smoke
[ -f $A/runs/SMOKE_OK ] || { echo "smoke FAILED: stopping before the runs" >> $L; touch $A/G_DONE; exit 1; }
# ---------------- G1: (a) the door finder alone; (b) enter flights, old and new code, same seeds, ideal operator; the G plans meanwhile
(timeout 2400 python test_door.py $A/runs/g1a.json 30 > $A/runs/g1a.log 2>&1; step g1a $?) &
G1A=$!
(wait $DL; timeout 2400 python eval_missions.py parse $A/models/parser_adapter $M1 $A/runs/g_plan.jsonl 300 G 6 > $A/runs/g_plan.log 2>&1; step g_plan $?
 timeout 1800 python eval_missions.py parse - - $A/runs/h_plan.jsonl 120 H 6 > $A/runs/h_plan.log 2>&1; step h_plan $?) &
PL=$!
ACTS_OPERATOR=1 timeout 2400 python test_actions.py $A/runs/g1b_new.jsonl 40 $WK enter > $A/runs/g1b_new.log 2>&1; step g1b_new $?
ACTS_OPERATOR=1 DOOR_OLD=1 timeout 2400 python test_actions.py $A/runs/g1b_old.jsonl 40 $WK enter > $A/runs/g1b_old.log 2>&1; step g1b_old $?
wait $G1A
python - >> $L 2>&1 <<'PY'
import json; A = '/home/ubuntu/af08/runs/'
ok = lambda f: sum(bool(r.get('steps') and r['steps'][0].get('ok')) for r in map(json.loads, open(A + f)))
n_new, n_old = ok('g1b_new.jsonl'), ok('g1b_old.jsonl')
try: s = json.load(open(A + 'g1a.json'))['summary']
except Exception: s = {}
closed_bad = (s.get('new') or {}).get('closed_false_door', 0)
keep = n_new >= n_old and closed_bad == 0
json.dump({'enter_new': n_new, 'enter_old': n_old, 'g1a': s, 'keep_new_door': keep}, open(A + 'g1_rule.json', 'w'), indent=1)
open('/home/ubuntu/af08/DOOR_ENV', 'w').write('0' if keep else '1'); print('G1 rule: new', n_new, 'old', n_old, 'closed false (new)', closed_bad, '-> keep new' if keep else '-> back to the old finder')
PY
export DOOR_OLD=$(cat $A/DOOR_ENV); echo "DOOR_OLD=$DOOR_OLD" >> $L; pack g1
wait $PL; pack plans
[ -s $A/runs/g_plan.jsonl ] || { echo "no G plan FAILED" >> $L; touch $A/G_DONE; exit 1; }
# ---------------- G2: the final test (full = alone; reflex = no core), then operator on the first 200
timeout 7200 python eval_missions.py fly $DET $ATT $A/runs/g_plan.jsonl $A/runs/g2.jsonl $WK full,reflex > $A/runs/g2.log 2>&1; step g2 $?; pack g2
head -200 $A/runs/g_plan.jsonl > $A/runs/g_plan200.jsonl
timeout 5400 python eval_missions.py fly $DET $ATT $A/runs/g_plan200.jsonl $A/runs/g2_operator.jsonl $WK operator > $A/runs/g2_operator.log 2>&1; step g2_operator $?; pack g2op
# ---------------- G3: slow feelings in hard conditions
timeout 6000 python eval_missions.py fly $DET $ATT $A/runs/h_plan.jsonl $A/runs/g3.jsonl $WK full,telemetry,slow_none,nolost,noenergy > $A/runs/g3.log 2>&1; step g3 $?; pack g3
# ---------------- G4: false alarm / delayed danger on the first 150 G missions
head -150 $A/runs/g_plan.jsonl > $A/runs/g_plan150.jsonl
timeout 3600 python eval_missions.py fly $DET $ATT $A/runs/g_plan150.jsonl $A/runs/g4.jsonl $WK false_alarm,delayed > $A/runs/g4.log 2>&1; step g4 $?; pack g4
touch $A/G_DONE; echo "G done $(date)" >> $L
