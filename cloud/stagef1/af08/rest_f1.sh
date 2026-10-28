#!/usr/bin/env bash
# The rest of the F1 full run after the parser hit its time limit (1 of 3 epochs done; best = epoch 0). Packs after every step.
set -u; A=$HOME/af08; L=$A/runs/setup.log; D=$A/data; M1=$HOME/models/qwen1.5b; M7=$HOME/models/qwen7b
export OSM2_DIR=$A/osm2 MUJOCO_GL=egl YOLO_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1; . $HOME/venv/bin/activate; cd $A/src
pack() { mkdir -p $A/runs/models; cp $A/runs/yolo/det/weights/best.pt $A/runs/models/detector.pt 2>/dev/null; cp $A/runs/attrs/attrs_best.pt $A/runs/models/ 2>/dev/null
  rm -rf $A/runs/models/parser_adapter; cp -r $A/runs/parser/best $A/runs/models/parser_adapter; cp $A/runs/attrs/attrs_log.json $A/runs/parser/train_log.json $A/runs/ 2>/dev/null
  cp $A/runs/yolo/det/results.csv $A/runs/yolo_results.csv 2>/dev/null; cd $A/runs && tar czf f1.tgz *.log *.json *.jsonl *.csv models 2>/dev/null; cd $A/src; echo "packed $1 $(date)" >> $L; }
echo "rest start $(date)" >> $L
timeout 1200 python train_attrs.py $D/percep $A/runs/attrs --epochs 12 > $A/runs/attrs.log 2>&1; echo "attrs rc=$? $(date)" >> $L; pack attrs
timeout 900 python eval_detector.py $A/runs/yolo/det/weights/best.pt $D/percep $A/runs/eval_det.json > $A/runs/eval_det.log 2>&1; echo "eval_det rc=$? $(date)" >> $L; pack eval_det
timeout 1500 python train_parser.py eval $A/runs/parser/best $D/parser_test.jsonl $A/runs/eval_parser.json --base $M1 > $A/runs/eval_parser.log 2>&1; echo "eval_parser rc=$? $(date)" >> $L; pack eval_parser
timeout 3600 python eval_grounding.py $A/runs/yolo/det/weights/best.pt $A/runs/attrs/attrs_best.pt $A/runs/parser/best $M1 $A/runs/eval_f2.json 60 > $A/runs/f2.log 2>&1; echo "f2 rc=$? $(date)" >> $L; pack f2
timeout 3000 python train_parser.py eval $M7 $D/parser_test.jsonl $A/runs/eval_parser_7b_fewshot.json --fewshot $D/parser_train.jsonl > $A/runs/eval_7b.log 2>&1; echo "eval_7b rc=$? $(date)" >> $L; pack eval_7b
touch $A/REST_DONE
