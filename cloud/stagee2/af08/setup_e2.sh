#!/usr/bin/env bash
# GPU VM (Ubuntu 24.04 CUDA, one RTX 3080 10 GB; the 7B with its language part in 4 bits, sec. 13.3): stage E zoom check (FEELINGS_DESIGN 13.3). 1) render the same 400 scenes with
# close-ups (E_ZOOM=4), check scenes.jsonl is byte-identical to step 1; 2) Qwen2.5-VL-7B without training: smoke on 16
# scenes (stop if > 10 % unparsed), then zoom|familiar, zoom|novel and a rerun of som|familiar, som|novel on this GPU. Marker E2_DONE; a failed
# step writes '... FAILED' to setup.log.
set -u; A=$HOME/af08; L=$A/runs/setup.log; mkdir -p $A/runs; echo "setup start $(date)" >> $L
fin() { cd $A/runs && tar czf e2.tgz sceneZ/scenes.jsonl sceneZ/zoom.jsonl sceneZ/img/e000*_som.png sceneZ/crop/e000* vlm*.json *.log 2> /dev/null; echo "packed $(date)" >> $L; touch $A/E2_DONE; }
nvidia-smi -L >> $L 2>&1 || { echo "no GPU FAILED" >> $L; fin; exit 1; }
df -h / | tail -1 >> $L; export PIP_NO_CACHE_DIR=1
P="timeout 3000 pip install -q --timeout 30 --retries 10"; MIR="-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq >> $L 2>&1
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip python3-dev build-essential libegl1 libgl1 libosmesa6 libglx-mesa0 >> $L 2>&1
python3 -m venv $HOME/venv >> $L 2>&1; . $HOME/venv/bin/activate; $P $MIR --upgrade pip >> $L 2>&1
M=$HOME/models/qwen2.5-vl-7b; G="from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-VL-7B-Instruct', local_dir='$M')"
$P $MIR huggingface_hub >> $L 2>&1
(timeout 2400 python -c "$G" >> $A/runs/dl.log 2>&1 || HF_ENDPOINT=https://hf-mirror.com python -c "$G" >> $A/runs/dl.log 2>&1; echo "7B download rc=$? $(du -sh $M | cut -f1) $(date)" >> $L) &
DL=$!
$P torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu126 >> $L 2>&1
$P $MIR transformers==4.57.6 accelerate bitsandbytes mujoco==3.17.0 PyOpenGL pillow numpy >> $L 2>&1
python -c "import torch,torchvision,transformers,mujoco,OpenGL,bitsandbytes;print('deps ok',torch.__version__,torch.cuda.is_available(),transformers.__version__,mujoco.__version__)" >> $L 2>&1 || { echo "deps FAILED" >> $L; fin; exit 1; }
cd $A/src; W=$(( $(nproc --all) - 1 )); export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MUJOCO_GL=egl; E_ZOOM=4 python stage_e_scenes.py 2 1 /tmp/gltest > $A/runs/gltest.log 2>&1 || { export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa; rm -rf /tmp/gltest; E_ZOOM=4 python stage_e_scenes.py 2 1 /tmp/gltest >> $A/runs/gltest.log 2>&1; }
ls /tmp/gltest/crop/e0000_1.png > /dev/null 2>&1 || { echo "render FAILED"; tail -5 $A/runs/gltest.log; } >> $L
ls /tmp/gltest/crop/e0000_1.png > /dev/null 2>&1 || { fin; exit 1; }
echo "render backend $MUJOCO_GL $(date)" >> $L
E_ZOOM=4 python stage_e_scenes.py 400 $W $A/runs/sceneZ > $A/runs/scenes.log 2>&1; echo "scenes rc=$? $(tail -1 $A/runs/scenes.log) $(date)" >> $L
echo "scenes identical to step 1: $(cmp -s $A/runs/sceneZ/scenes.jsonl $A/scenes_step1.jsonl && echo yes || echo NO) $(ls $A/runs/sceneZ/crop | wc -l) close-ups" >> $L
wait $DL; ls $M/*.safetensors > /dev/null 2>&1 || { echo "model FAILED" >> $L; fin; exit 1; }
export OMP_NUM_THREADS=4
BS=4; python -u stage_e_vlm.py --data $A/runs/sceneZ --model $M --out $A/runs/vlm_zoom_smoke.json --limit 16 --conds 'zoom|familiar,zoom|novel' --quant4 --bs $BS > $A/runs/vlm_zoom_smoke.log 2>&1; echo "zoom smoke rc=$? bs=$BS $(date) $(grep -m1 'GPU memory' $A/runs/vlm_zoom_smoke.log)" >> $L
if ! ls $A/runs/vlm_zoom_smoke.json > /dev/null 2>&1; then tail -3 $A/runs/vlm_zoom_smoke.log >> $L; BS=1; python -u stage_e_vlm.py --data $A/runs/sceneZ --model $M --out $A/runs/vlm_zoom_smoke.json --limit 16 --conds 'zoom|familiar,zoom|novel' --quant4 --bs $BS > $A/runs/vlm_zoom_smoke.log 2>&1; echo "zoom smoke retry rc=$? bs=$BS $(date)" >> $L; fi
F=$(python -c "import json; V=json.load(open('$A/runs/vlm_zoom_smoke.json'))['conds']; a=[x['pick'] for r in V.values() for x in r['answers'].values()]; print(round(sum(p == 'fail' for p in a) / len(a), 3))" 2>> $L || echo 1)
echo "zoom smoke unparsed share $F" >> $L
python -c "import sys; sys.exit(0 if float('$F') <= .1 else 1)" || { echo "zoom smoke parse FAILED" >> $L; fin; exit 1; }
python -u stage_e_vlm.py --data $A/runs/sceneZ --model $M --out $A/runs/vlm_zoom.json --conds 'zoom|familiar,zoom|novel,som|familiar,som|novel' --quant4 --bs $BS > $A/runs/vlm_zoom.log 2>&1; echo "vlm rc=$? $(date)" >> $L
fin
