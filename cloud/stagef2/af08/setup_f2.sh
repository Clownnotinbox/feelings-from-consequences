#!/usr/bin/env bash
# drone-f2 (RTX 3090, 32 vCPU): (1) F3b again (MISSIONS_DESIGN_RU.md 6.2: 7B paraphrases with a back-check -> the 1.5B parser
# retrained -> F3 test set + 100 hand-written) — its first run was lost with drone-f1; (2) in parallel, debug flights of every
# action with the current brain (test_actions.py, trained perception, true parse). Packs after every step; a local waiter pulls.
set -u; A=$HOME/af08; L=$A/runs/setup.log; D=$A/data; mkdir -p $A/runs $D; echo "setup start $(date)" >> $L
export OSM2_DIR=$A/osm2 MUJOCO_GL=egl PIP_NO_CACHE_DIR=1 YOLO_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { echo "$1 rc=$2 $(date)" >> $L; [ "$2" != "0" ] && { echo "$1 FAILED" >> $L; tail -20 $A/runs/$1.log >> $L 2>/dev/null; }; return 0; }
pack() { cd $A/runs && tar czf f2.tgz *.log *.json *.jsonl models3b 2>/dev/null; cd $A/src; echo "packed $1 $(date)" >> $L; }
nvidia-smi -L >> $L 2>&1 || { echo "no GPU FAILED" >> $L; touch $A/F2_DONE; exit 1; }
P="timeout 3000 pip install -q --timeout 30 --retries 10"; MIR="-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq >> $L 2>&1
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip python3-dev build-essential libegl1 libgl1 libosmesa6 libglx-mesa0 libglib2.0-0 libgomp1 >> $L 2>&1
python3 -m venv $HOME/venv >> $L 2>&1; . $HOME/venv/bin/activate; $P $MIR --upgrade pip >> $L 2>&1; $P $MIR huggingface_hub >> $L 2>&1
M1=$HOME/models/qwen1.5b; M7=$HOME/models/qwen7b
dl() { (timeout 1500 python -c "from huggingface_hub import snapshot_download as s; s('$1', local_dir='$2')" >> $A/runs/dl.log 2>&1 || HF_ENDPOINT=https://hf-mirror.com timeout 2400 python -c "from huggingface_hub import snapshot_download as s; s('$1', local_dir='$2')" >> $A/runs/dl.log 2>&1); echo "download $1 rc=$? $(du -sh $2 | cut -f1) $(date)" >> $L; }
(dl Qwen/Qwen2.5-1.5B-Instruct $M1; dl Qwen/Qwen2.5-7B-Instruct $M7) &
DL=$!
$P torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu126 >> $L 2>&1
timeout 1200 pip install -q --timeout 20 --retries 3 transformers==4.57.6 accelerate peft mujoco==3.17.0 PyOpenGL pillow numpy shapely mapbox_earcut scipy ultralytics open3d scikit-learn==1.9.1 >> $L 2>&1 || $P $MIR transformers==4.57.6 accelerate peft mujoco==3.17.0 PyOpenGL pillow numpy shapely mapbox_earcut scipy ultralytics open3d scikit-learn==1.9.1 >> $L 2>&1  # PyPI first (the aliyun mirror stalled on 27.10)
python -c "import torch,transformers,peft,mujoco,ultralytics,shapely,mapbox_earcut,open3d,sklearn;print('deps ok',torch.__version__,torch.cuda.is_available(),mujoco.__version__,sklearn.__version__)" >> $L 2>&1 || { echo "deps FAILED" >> $L; touch $A/F2_DONE; exit 1; }
cd $A/src; W=$(( $(nproc --all) - 2 ))
# ---------------- SMOKE (seconds): the flight stack and the data generator
timeout 600 python -c "
import numpy as np, city3d as C, missions as M, mission_env as ME, safety as SF, perceive as PC
W0, _, _ = C.build('akadem_line', 1, actors=False); mis = M.make(W0, np.random.default_rng(1)); env = ME.MissionEnv('akadem_line', 1, mis, render=False, cam=(160, 120), gimbal=(960, 720)); env.reset()
s = SF.Safety(env); o = env.step({'vx': 0, 'vy': 0, 'vz': 1, 'yaw_rate': 0}); s(o, {'vx': 0, 'vy': 0, 'vz': 1, 'yaw_rate': 0}); p = PC.Perceiver('$A/models/detector.pt', '$A/models/attrs_best.pt')
env.set_gimbal(0, -45); rgb, dep, cam = env.gimbal_frame(); print('smoke ok', len(p(rgb, dep, cam, env.fovy)))
" > $A/runs/smoke_stack.log 2>&1; step smoke_stack $?
# ---------------- (2) flights of every action, in the background (CPU), from now on
(timeout 7200 python test_actions.py $A/runs/acts5.jsonl 6 20 > $A/runs/acts5.log 2>&1; step acts5 $?; pack acts5) &
FL=$!
# ---------------- (1) F3b
python parser_data.py $D/parser_train.jsonl train 1500 8 > $A/runs/pdata.log 2>&1 && python parser_data.py $D/parser_val.jsonl val 200 8 >> $A/runs/pdata.log 2>&1 && python parser_data.py $D/parser_test.jsonl test 300 8 >> $A/runs/pdata.log 2>&1; step pdata $?
wait $DL
timeout 2700 python paraphrase_data.py $M7 $D/parser_train.jsonl $D/para_train.jsonl 6000 1 > $A/runs/f3b_para.log 2>&1; step f3b_para $?
timeout 600 python paraphrase_data.py $M7 $D/parser_val.jsonl $D/para_val.jsonl 400 2 > $A/runs/f3b_para_val.log 2>&1; step f3b_para_val $?
cp $D/para_train.jsonl $D/para_val.jsonl $A/runs/ 2>/dev/null; pack para
python - > $A/runs/f3b_mix.log 2>&1 <<'EOF'
import json, random
D = '/home/ubuntu/af08/data'; R = random.Random(0); rd = lambda p: [json.loads(l) for l in open(p, encoding='utf-8')]
tr, va, ptr, pva = rd(f'{D}/parser_train.jsonl'), rd(f'{D}/parser_val.jsonl'), rd(f'{D}/para_train.jsonl'), rd(f'{D}/para_val.jsonl')
R.shuffle(tr); R.shuffle(va); mix = tr[:5000] + ptr; mv = va[:200] + pva; R.shuffle(mix); R.shuffle(mv)
for p, rows in ((f'{D}/mix_train.jsonl', mix), (f'{D}/mix_val.jsonl', mv)):
    with open(p, 'w', encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
st = {'train_template': 5000, 'train_paraphrases': len(ptr), 'val_template': 200, 'val_paraphrases': len(pva)}
json.dump(st, open('/home/ubuntu/af08/runs/para_stats.json', 'w')); print(st)
EOF
step f3b_mix $?
timeout 5400 python train_parser.py train $D/mix_train.jsonl $D/mix_val.jsonl $A/runs/parser3b $M1 --epochs 2 --bs 8 --accum 2 > $A/runs/f3b_train.log 2>&1; step f3b_train $?
mkdir -p $A/runs/models3b; [ -d $A/runs/parser3b/best ] && cp -r $A/runs/parser3b/best $A/runs/models3b/parser_adapter; cp $A/runs/parser3b/train_log.json $A/runs/parser3b_train_log.json 2>/dev/null; pack train
timeout 1800 python train_parser.py eval $A/runs/parser3b/best $D/parser_test.jsonl $A/runs/eval_parser_3b.json --base $M1 > $A/runs/f3b_eval.log 2>&1; step f3b_eval $?; pack eval
wait $FL; pack all; touch $A/F2_DONE
