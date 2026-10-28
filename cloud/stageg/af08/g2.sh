#!/usr/bin/env bash
# Stage G resumed (28.10, 00:26): the first G1 run ran out of GPU memory (26 flight workers x 0.9 GB + the parser + the PX4 check
# on one 24 GB card), its rows are discarded. Order now: the plans first (the parser alone on the GPU), then flights with 20
# workers (nice 5, so the real-time PX4 check keeps its pace), the PX4 check started after the plans. Setup and smoke are done.
set -u; A=$HOME/af08; L=$A/runs/setup.log; echo "resume start $(date)" >> $L
export OSM2_DIR=$A/osm2 MUJOCO_GL=egl YOLO_OFFLINE=1 PYTHONIOENCODING=utf-8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1
step() { echo "$1 rc=$2 $(date)" >> $L; [ "$2" != "0" ] && { echo "$1 FAILED" >> $L; tail -25 $A/runs/$1.log >> $L 2>/dev/null; }; return 0; }
pack() { cd $A/runs && tar czf g.tgz *.log *.json *.jsonl 2>/dev/null; cd $A/src; echo "packed $1 $(date)" >> $L; }
. $HOME/venv/bin/activate; cd $A/src; WK=20; DET=$A/models/detector.pt; ATT=$A/models/attrs_best.pt; M1=$HOME/models/qwen1.5b; F="nice -n 5"
timeout 2400 python eval_missions.py parse $A/models/parser_adapter $M1 $A/runs/g_plan.jsonl 300 G 24 > $A/runs/g_plan.log 2>&1; step g_plan $?
timeout 1800 python eval_missions.py parse - - $A/runs/h_plan.jsonl 120 H 24 > $A/runs/h_plan.log 2>&1; step h_plan $?; pack plans
[ -s $A/runs/g_plan.jsonl ] || { echo "no G plan FAILED" >> $L; touch $A/G_DONE; exit 1; }
(cd $A && setsid nohup bash px4_run.sh > runs/px4_run.out 2>&1 < /dev/null &)  # the PX4 check (3 GPU processes) beside the flights
(timeout 2400 python test_door.py $A/runs/g1a.json 30 > $A/runs/g1a.log 2>&1; step g1a $?) &
G1A=$!
ACTS_OPERATOR=1 timeout 2400 $F python test_actions.py $A/runs/g1b_new.jsonl 40 $WK enter > $A/runs/g1b_new.log 2>&1; step g1b_new $?
ACTS_OPERATOR=1 DOOR_OLD=1 timeout 2400 $F python test_actions.py $A/runs/g1b_old.jsonl 40 $WK enter > $A/runs/g1b_old.log 2>&1; step g1b_old $?
wait $G1A
python - >> $L 2>&1 <<'PY'
import json; A = '/home/ubuntu/af08/runs/'
rows = lambda f: [json.loads(l) for l in open(A + f)]
ok = lambda R: sum(bool(r.get('steps') and r['steps'][0].get('ok')) for r in R)
new, old = rows('g1b_new.jsonl'), rows('g1b_old.jsonl'); n_new, n_old = ok(new), ok(old); err = sum('error' in r for r in new + old)
try: s = json.load(open(A + 'g1a.json'))['summary']
except Exception: s = {}
closed_bad = (s.get('new') or {}).get('closed_false_door', 0)
keep = n_new >= n_old and closed_bad == 0
json.dump({'enter_new': n_new, 'enter_old': n_old, 'errors': err, 'g1a': s, 'keep_new_door': keep}, open(A + 'g1_rule.json', 'w'), indent=1)
open('/home/ubuntu/af08/DOOR_ENV', 'w').write('0' if keep else '1'); print('G1 rule: new', n_new, 'old', n_old, 'errors', err, 'closed false (new)', closed_bad, '-> keep new' if keep else '-> back to the old finder')
PY
export DOOR_OLD=$(cat $A/DOOR_ENV); echo "DOOR_OLD=$DOOR_OLD" >> $L; pack g1
timeout 7200 $F python eval_missions.py fly $DET $ATT $A/runs/g_plan.jsonl $A/runs/g2.jsonl $WK full,reflex > $A/runs/g2.log 2>&1; step g2 $?; pack g2
head -200 $A/runs/g_plan.jsonl > $A/runs/g_plan200.jsonl
timeout 5400 $F python eval_missions.py fly $DET $ATT $A/runs/g_plan200.jsonl $A/runs/g2_operator.jsonl $WK operator > $A/runs/g2_operator.log 2>&1; step g2_operator $?; pack g2op
timeout 6000 $F python eval_missions.py fly $DET $ATT $A/runs/h_plan.jsonl $A/runs/g3.jsonl $WK full,telemetry,slow_none,nolost,noenergy > $A/runs/g3.log 2>&1; step g3 $?; pack g3
head -150 $A/runs/g_plan.jsonl > $A/runs/g_plan150.jsonl
timeout 3600 $F python eval_missions.py fly $DET $ATT $A/runs/g_plan150.jsonl $A/runs/g4.jsonl $WK false_alarm,delayed > $A/runs/g4.log 2>&1; step g4 $?; pack g4
touch $A/G_DONE; echo "G done $(date)" >> $L
