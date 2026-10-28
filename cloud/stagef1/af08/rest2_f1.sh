#!/usr/bin/env bash
# F2 again (crashed on a numpy value in the parse comparison, fixed), then the 7B baseline with what time is left. Packs after each.
set -u; A=$HOME/af08; L=$A/runs/setup.log; D=$A/data; M1=$HOME/models/qwen1.5b; M7=$HOME/models/qwen7b
export OSM2_DIR=$A/osm2 MUJOCO_GL=egl YOLO_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1; . $HOME/venv/bin/activate; cd $A/src
pack() { cd $A/runs && tar czf f1.tgz *.log *.json *.jsonl *.csv models 2>/dev/null; cd $A/src; echo "packed $1 $(date)" >> $L; }
echo "rest2 start $(date)" >> $L; mv $A/runs/f2.log $A/runs/f2_crash1.log
timeout 3000 python eval_grounding.py $A/runs/yolo/det/weights/best.pt $A/runs/attrs/attrs_best.pt $A/runs/parser/best $M1 $A/runs/eval_f2.json 60 > $A/runs/f2.log 2>&1; echo "f2 rc=$? $(date)" >> $L; pack f2
timeout 2400 python train_parser.py eval $M7 $D/parser_test.jsonl $A/runs/eval_parser_7b_fewshot.json --fewshot $D/parser_train.jsonl > $A/runs/eval_7b.log 2>&1; echo "eval_7b rc=$? $(date)" >> $L; pack eval_7b
touch $A/REST_DONE
