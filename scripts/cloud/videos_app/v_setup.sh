#!/usr/bin/env bash
# Setup only (venv, models) for the app checks and walkthroughs; then the app runs: test ru, test en, walk ru, walk en. Take 4 (28.10): the four core / no-core pairs of G2 flown again from the plan and composed side by side,
# setup as in stage G. Writes runs/videos_v3/, packs runs/v.tgz, touches V_DONE. The app walkthrough is v_app.sh (after SETUP_OK).
set -u; A=$HOME/af08; L=$A/runs/v.log; O=$A/runs/videos_v3; mkdir -p $O; echo "start $(date)" >> $L
export OSM2_DIR=$A/osm2 MUJOCO_GL=egl PIP_NO_CACHE_DIR=1 YOLO_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONIOENCODING=utf-8
fin() { touch $A/V_DONE; }
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
wait $DL; echo "qwen ready $(date)" >> $L; touch $A/SETUP_OK
for m in "test ru" "test en" "walk ru" "walk en"; do bash $A/v_app.sh $m; done
cd $A/runs && tar czf app_all.tgz apptest_ru apptest_en walk_ru/walkthrough.json walk_ru/app_flight.png walk_en/walkthrough.json walk_en/app_flight.png videos_v3/walkthrough_ru.mp4 videos_v3/walkthrough_en.mp4 *.log 2>/dev/null; touch $A/ALL_DONE
