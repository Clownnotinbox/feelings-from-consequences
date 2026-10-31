#!/usr/bin/env bash
# GPU VM (Ubuntu 24.04 CUDA, one RTX 3090/4090): stage E step 1 (FEELINGS_DESIGN 13.1). 1) render 400 scenes (MuJoCo, EGL;
# OSMesa if EGL fails); 2) the sentence-encoder rival; 3) Qwen2.5-VL-7B without training: smoke on 16 scenes, then
# 400 scenes x {point, som} x {familiar, novel}. Marker E1_DONE; a failed step writes '... FAILED' to setup.log.
set -u; A=$HOME/af08; L=$A/runs/setup.log; mkdir -p $A/runs; echo "setup start $(date)" >> $L
fin() { cd $A/runs && tar czf e1.tgz sceneE/scenes.jsonl sceneE/img/e000* sceneE/img/e001* embed.json vlm*.json *.log 2> /dev/null; echo "packed $(date)" >> $L; touch $A/E1_DONE; }
nvidia-smi -L >> $L 2>&1 || { echo "no GPU FAILED" >> $L; fin; exit 1; }
df -h / | tail -1 >> $L; export PIP_NO_CACHE_DIR=1
P="timeout 3000 pip install -q --timeout 30 --retries 10"; MIR="-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq >> $L 2>&1
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip python3-dev build-essential libegl1 libgl1 libosmesa6 libglx-mesa0 >> $L 2>&1
python3 -m venv $HOME/venv >> $L 2>&1; . $HOME/venv/bin/activate; $P $MIR --upgrade pip >> $L 2>&1
$P torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu126 >> $L 2>&1
$P $MIR transformers==4.57.6 accelerate sentence-transformers huggingface_hub mujoco==3.17.0 PyOpenGL pillow numpy >> $L 2>&1
python -c "import torch,torchvision,transformers,mujoco,OpenGL,sentence_transformers;print('deps ok',torch.__version__,torch.cuda.is_available(),transformers.__version__,mujoco.__version__)" >> $L 2>&1 || { echo "deps FAILED" >> $L; fin; exit 1; }
M=$HOME/models/qwen2.5-vl-7b; G="from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-VL-7B-Instruct', local_dir='$M')"
ME=$HOME/models/minilm; GE="from huggingface_hub import snapshot_download as s; s('sentence-transformers/all-MiniLM-L6-v2', local_dir='$ME')"
(timeout 2400 python -c "$G" >> $A/runs/dl.log 2>&1 || HF_ENDPOINT=https://hf-mirror.com python -c "$G" >> $A/runs/dl.log 2>&1; echo "7B download rc=$? $(du -sh $M | cut -f1) $(date)" >> $L) &
DL=$!
cd $A/src; W=$(( $(nproc --all) - 1 )); export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MUJOCO_GL=egl; python stage_e_scenes.py 2 1 /tmp/gltest > $A/runs/gltest.log 2>&1 || { export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa; rm -rf /tmp/gltest; python stage_e_scenes.py 2 1 /tmp/gltest >> $A/runs/gltest.log 2>&1; }
ls /tmp/gltest/img/e0000.png > /dev/null 2>&1 || { echo "render FAILED" >> $L; fin; exit 1; }
echo "render backend $MUJOCO_GL $(date)" >> $L
python stage_e_scenes.py 400 $W $A/runs/sceneE > $A/runs/scenes.log 2>&1; echo "scenes rc=$? $(tail -1 $A/runs/scenes.log) $(date)" >> $L
(timeout 900 python -c "$GE" >> $A/runs/dl.log 2>&1 || HF_ENDPOINT=https://hf-mirror.com python -c "$GE" >> $A/runs/dl.log 2>&1)
python stage_e_vlm.py --data $A/runs/sceneE --embed $ME --out $A/runs/embed.json > $A/runs/embed.log 2>&1; echo "embed rc=$? $(date)" >> $L
wait $DL; ls $M/*.safetensors > /dev/null 2>&1 || { echo "model FAILED" >> $L; fin; exit 1; }
export OMP_NUM_THREADS=4
python -u stage_e_vlm.py --data $A/runs/sceneE --model $M --out $A/runs/vlm_smoke.json --limit 16 > $A/runs/vlm_smoke.log 2>&1; echo "vlm smoke rc=$? $(date)" >> $L
python -u stage_e_vlm.py --data $A/runs/sceneE --model $M --out $A/runs/vlm7b.json > $A/runs/vlm7b.log 2>&1; echo "vlm rc=$? $(date)" >> $L
fin
