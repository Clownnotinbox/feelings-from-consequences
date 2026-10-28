#!/usr/bin/env bash
# Supplementary videos, take 3 (28.10): the four core / no-core pairs of G2 flown again from the plan and composed side by side,
# setup as in stage G. Writes runs/videos_v3/, packs runs/v.tgz, touches V_DONE. The app walkthrough is v_app.sh (after SETUP_OK).
set -u; A=$HOME/af08; L=$A/runs/v.log; O=$A/runs/videos_v3; mkdir -p $O; echo "start $(date)" >> $L
export OSM2_DIR=$A/osm2 MUJOCO_GL=egl PIP_NO_CACHE_DIR=1 YOLO_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONIOENCODING=utf-8
fin() { cd $A/runs && tar czf v.tgz *.log check_v.json videos_v3 2>/dev/null; echo "packed $(date)" >> $L; touch $A/V_DONE; }
nvidia-smi -L >> $L 2>&1 || { echo "no GPU FAILED" >> $L; fin; exit 1; }
P="timeout 3000 pip install -q --timeout 30 --retries 10"; MIR="-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq >> $L 2>&1
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip python3-dev build-essential libegl1 libgl1 libosmesa6 libglx-mesa0 libglib2.0-0 libgomp1 ffmpeg curl >> $L 2>&1
python3 -m venv $HOME/venv >> $L 2>&1; . $HOME/venv/bin/activate; $P $MIR --upgrade pip >> $L 2>&1; $P $MIR huggingface_hub >> $L 2>&1
M1=$A/models/qwen1.5b
(timeout 1500 python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-1.5B-Instruct', local_dir='$M1')" >> $A/runs/dl.log 2>&1 || HF_ENDPOINT=https://hf-mirror.com timeout 2400 python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-1.5B-Instruct', local_dir='$M1')" >> $A/runs/dl.log 2>&1; echo "download qwen1.5b rc=$? $(du -sh $M1 | cut -f1) $(date)" >> $L) &
DL=$!
$P torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu126 >> $L 2>&1
timeout 1200 pip install -q --timeout 20 --retries 3 transformers==4.57.6 accelerate peft mujoco==3.17.0 PyOpenGL pillow numpy shapely mapbox_earcut scipy ultralytics open3d scikit-learn==1.9.1 playwright >> $L 2>&1 || $P $MIR transformers==4.57.6 accelerate peft mujoco==3.17.0 PyOpenGL pillow numpy shapely mapbox_earcut scipy ultralytics open3d scikit-learn==1.9.1 playwright >> $L 2>&1
timeout 900 python -m playwright install --with-deps chromium >> $L 2>&1
python -c "import torch,transformers,peft,mujoco,ultralytics,shapely,open3d,sklearn,playwright;print('deps ok',torch.__version__,torch.cuda.is_available(),mujoco.__version__)" >> $L 2>&1 || { echo "deps FAILED" >> $L; fin; exit 1; }
cd $A/src
# 1. the eight flights, six at a time (about 1 GB of video memory each; the app test shares the 10 GB card)
printf "%s\n" "42 full" "42 reflex" "45 full" "45 reflex" "112 full" "112 reflex" "206 full" "206 reflex" | OMP_NUM_THREADS=2 xargs -P 6 -L 1 sh -c "nice -n 10 timeout 2700 python render_flight.py $A/runs/g_plan_full.jsonl \$0 \$1 $A/runs/vr 2 >> $A/runs/render.log 2>&1" &
REN=$!
touch $A/SETUP_OK; wait $DL; echo "qwen ready $(date)" >> $L
wait $REN; echo "renders done $(ls $A/runs/vr | tr '\n' ' ') $(date)" >> $L
# 3. side-by-side videos and stills for the paper
for i in 42 45 112 206; do
  F=$A/runs/vr/G${i}_full; R=$A/runs/vr/G${i}_reflex
  [ -f $F/result.json ] && [ -f $R/result.json ] || { echo "G$i missing a render" >> $L; continue; }
  for LG in ru en; do
    timeout 1800 python compose_compare.py $F $R $O/G${i}_compare_$LG.mp4 $LG 25 >> $A/runs/compose.log 2>&1
    python compose_compare.py $F $R x $LG --still $O/G${i}_compare_$LG.jpg >> $A/runs/compose.log 2>&1
  done
done
# 4. does each replay end as its G2 flight did?
python - >> $L 2>&1 <<'PY'
import json, os
A = '/home/ubuntu/af08/runs/'
G = {(r['i'], r['variant']): r for r in map(json.loads, open(A + 'g2.jsonl')) if r['i'] in (42, 45, 112, 206)}
out = {}
for (i, v), g in sorted(G.items()):
    p = f'{A}vr/G{i}_{v}/result.json'
    if not os.path.exists(p):
        out[f'{i}_{v}'] = 'missing'; continue
    r = json.load(open(p))
    rep = {'steps_ok': [bool(s.get('ok')) for s in r['steps']], 'static': r['collisions'] - r['bird_hits']}
    out[f'{i}_{v}'] = {'g2': {'steps_ok': g['steps_ok'], 'static': g['hits_static']}, 'replay': rep,
                       'same': rep['steps_ok'] == g['steps_ok'] and rep['static'] == g['hits_static']}
json.dump(out, open(A + 'check_v.json', 'w'), indent=1); print('check', {k: (v if isinstance(v, str) else v['same']) for k, v in out.items()})
PY
fin
