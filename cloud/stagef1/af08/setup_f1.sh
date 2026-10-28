#!/usr/bin/env bash
# GPU VM (Ubuntu 24.04 CUDA, RTX 3090): stage F step 1 (MISSIONS_DESIGN_RU.md): perception data -> detector (YOLO) + attribute
# net; mission texts -> parser (Qwen2.5-1.5B LoRA) + zero-shot 7B baseline; checks F1, F2, F3. First a SMOKE run of every
# part on tiny data (stops on any failure), then the full runs. Markers in runs/setup.log; '... FAILED' on failure; F1_DONE at the end.
set -u; A=$HOME/af08; L=$A/runs/setup.log; mkdir -p $A/runs $A/data; echo "setup start $(date)" >> $L
export OSM2_DIR=$A/osm2 MUJOCO_GL=egl PIP_NO_CACHE_DIR=1 YOLO_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
fin() { cd $A/runs && tar czf f1.tgz *.log *.json *.jsonl models 2> /dev/null; echo "packed $(date)" >> $L; touch $A/F1_DONE; }
step() { echo "$1 rc=$2 $(date)" >> $L; if [ "$2" != "0" ]; then echo "$1 FAILED" >> $L; tail -25 $A/runs/$1.log >> $L 2>/dev/null; fin; exit 1; fi; }
nvidia-smi -L >> $L 2>&1 || { echo "no GPU FAILED" >> $L; fin; exit 1; }
df -h / | tail -1 >> $L
P="timeout 3000 pip install -q --timeout 30 --retries 10"; MIR="-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq >> $L 2>&1
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip python3-dev build-essential libegl1 libgl1 libosmesa6 libglx-mesa0 libglib2.0-0 >> $L 2>&1
python3 -m venv $HOME/venv >> $L 2>&1; . $HOME/venv/bin/activate; $P $MIR --upgrade pip >> $L 2>&1; $P $MIR huggingface_hub >> $L 2>&1
M1=$HOME/models/qwen1.5b; M7=$HOME/models/qwen7b
dl() { (timeout 1500 python -c "from huggingface_hub import snapshot_download as s; s('$1', local_dir='$2')" >> $A/runs/dl.log 2>&1 || HF_ENDPOINT=https://hf-mirror.com timeout 2400 python -c "from huggingface_hub import snapshot_download as s; s('$1', local_dir='$2')" >> $A/runs/dl.log 2>&1); echo "download $1 rc=$? $(du -sh $2 | cut -f1) $(date)" >> $L; }
(dl Qwen/Qwen2.5-1.5B-Instruct $M1; dl Qwen/Qwen2.5-7B-Instruct $M7) &
DL=$!
$P torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu126 >> $L 2>&1
$P $MIR transformers==4.57.6 accelerate peft mujoco==3.17.0 PyOpenGL pillow numpy shapely mapbox_earcut scipy ultralytics >> $L 2>&1
python -c "import torch,torchvision,transformers,peft,mujoco,ultralytics,shapely,mapbox_earcut;print('deps ok',torch.__version__,torch.cuda.is_available(),transformers.__version__,mujoco.__version__,ultralytics.__version__)" >> $L 2>&1 || { echo "deps FAILED" >> $L; fin; exit 1; }
cd $A/src; W=$(( $(nproc --all) - 1 ))
# pretrained detector weights (COCO): GitHub, else the HF mirror; without them the detector trains from scratch (recorded)
YW=$A/yolo11s.pt
timeout 120 python -c "import urllib.request as u; u.urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt', '$YW')" >> $L 2>&1 \
  || timeout 300 python -c "import urllib.request as u; u.urlretrieve('https://hf-mirror.com/Ultralytics/YOLO11/resolve/main/yolo11s.pt', '$YW')" >> $L 2>&1
cp $YW $A/src/yolo11n.pt 2> /dev/null  # the AMP self-check looks for yolo11n.pt; give it a local file instead of a download
ls -la $YW >> $L 2>&1 && echo "yolo weights ok" >> $L || echo "yolo weights missing: train from scratch" >> $L
YM=$( [ -s $YW ] && echo $YW || echo yolo11s.yaml )
# ---------------- SMOKE: every part on tiny data
S=$A/smoke; rm -rf $S; mkdir -p $S
python percep_data.py $S/percep train 1 3 2 > $A/runs/smoke_percep.log 2>&1 && python percep_data.py $S/percep val 1 3 2 >> $A/runs/smoke_percep.log 2>&1 && python percep_data.py $S/percep test 1 3 2 >> $A/runs/smoke_percep.log 2>&1; step smoke_percep $?
python parser_data.py $S/ptrain.jsonl train 16 4 > $A/runs/smoke_pdata.log 2>&1 && python parser_data.py $S/pval.jsonl val 16 2 >> $A/runs/smoke_pdata.log 2>&1 && python parser_data.py $S/ptest.jsonl test 16 2 >> $A/runs/smoke_pdata.log 2>&1; step smoke_pdata $?
timeout 900 yolo detect train data=$S/percep/data.yaml model=$YM imgsz=640 epochs=1 batch=4 workers=2 plots=False project=$S/yolo name=det exist_ok=True > $A/runs/smoke_yolo.log 2>&1; step smoke_yolo $?
timeout 600 python train_attrs.py $S/percep $S/attrs --epochs 1 > $A/runs/smoke_attrs.log 2>&1; step smoke_attrs $?
timeout 600 python eval_detector.py $S/yolo/det/weights/best.pt $S/percep $S/eval_det.json > $A/runs/smoke_evdet.log 2>&1; step smoke_evdet $?
wait $DL; ls $M1/*.safetensors > /dev/null 2>&1 || { echo "model 1.5B FAILED" >> $L; fin; exit 1; }
timeout 900 python train_parser.py train $S/ptrain.jsonl $S/pval.jsonl $S/parser $M1 --epochs 1 --limit 32 > $A/runs/smoke_parser.log 2>&1; step smoke_parser $?
timeout 900 python train_parser.py eval $S/parser/best $S/ptest.jsonl $S/eval_parser.json --base $M1 --limit 8 > $A/runs/smoke_evparser.log 2>&1; step smoke_evparser $?
timeout 900 python eval_grounding.py $S/yolo/det/weights/best.pt $S/attrs/attrs_best.pt $S/parser/best $M1 $S/eval_f2.json 1 > $A/runs/smoke_f2.log 2>&1; step smoke_f2 $?
echo "SMOKE OK $(date)" >> $L
# ---------------- FULL
D=$A/data
python percep_data.py $D/percep train 30 40 $W > $A/runs/percep_train.log 2>&1; step percep_train $?
python percep_data.py $D/percep val 8 40 $W > $A/runs/percep_val.log 2>&1; step percep_val $?
python percep_data.py $D/percep test 8 40 $W > $A/runs/percep_test.log 2>&1; step percep_test $?
python parser_data.py $D/parser_train.jsonl train 1500 $W > $A/runs/pdata.log 2>&1 && python parser_data.py $D/parser_val.jsonl val 200 $W >> $A/runs/pdata.log 2>&1 && python parser_data.py $D/parser_test.jsonl test 300 $W >> $A/runs/pdata.log 2>&1; step pdata $?
timeout 5400 yolo detect train data=$D/percep/data.yaml model=$YM imgsz=960 epochs=25 batch=16 workers=6 patience=8 plots=False cos_lr=True project=$A/runs/yolo name=det exist_ok=True > $A/runs/yolo.log 2>&1 &
YP=$!
timeout 4800 python train_parser.py train $D/parser_train.jsonl $D/parser_val.jsonl $A/runs/parser $M1 --epochs 3 > $A/runs/parser.log 2>&1; step parser $?
timeout 2400 python train_parser.py eval $A/runs/parser/best $D/parser_test.jsonl $A/runs/eval_parser.json --base $M1 > $A/runs/eval_parser.log 2>&1; step eval_parser $?
ls $M7/*.safetensors > /dev/null 2>&1 && { timeout 3000 python train_parser.py eval $M7 $D/parser_test.jsonl $A/runs/eval_parser_7b_fewshot.json --fewshot $D/parser_train.jsonl > $A/runs/eval_7b.log 2>&1; echo "eval_7b rc=$? $(date)" >> $L; } || echo "7B missing: no zero-shot baseline" >> $L
wait $YP; step yolo $?
timeout 2400 python train_attrs.py $D/percep $A/runs/attrs --epochs 12 > $A/runs/attrs.log 2>&1; step attrs $?
timeout 2400 python eval_detector.py $A/runs/yolo/det/weights/best.pt $D/percep $A/runs/eval_det.json > $A/runs/eval_det.log 2>&1; step eval_det $?
timeout 4800 python eval_grounding.py $A/runs/yolo/det/weights/best.pt $A/runs/attrs/attrs_best.pt $A/runs/parser/best $M1 $A/runs/eval_f2.json 60 > $A/runs/f2.log 2>&1; step f2 $?
mkdir -p $A/runs/models; cp $A/runs/yolo/det/weights/best.pt $A/runs/models/detector.pt; cp $A/runs/attrs/attrs_best.pt $A/runs/models/; cp -r $A/runs/parser/best $A/runs/models/parser_adapter
cp $A/runs/attrs/attrs_log.json $A/runs/parser/train_log.json $A/runs/ 2> /dev/null; cp $A/runs/yolo/det/results.csv $A/runs/yolo_results.csv 2> /dev/null
fin
