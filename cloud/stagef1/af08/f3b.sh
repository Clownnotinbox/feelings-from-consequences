#!/usr/bin/env bash
# F3b (MISSIONS_DESIGN_RU.md 6.2): 7B paraphrases with a back-check -> retrain the 1.5B parser on templates + paraphrases ->
# evaluate on the same test set as F3 (test places, familiar / novel wording) + the 100 hand-written texts. Packs after each step.
set -u; A=$HOME/af08; L=$A/runs/setup.log; D=$A/data; M1=$HOME/models/qwen1.5b; M7=$HOME/models/qwen7b
export OSM2_DIR=$A/osm2 HF_HUB_DISABLE_TELEMETRY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; . $HOME/venv/bin/activate; cd $A/src
pack() { mkdir -p $A/runs/models3b; [ -d $A/runs/parser3b/best ] && { rm -rf $A/runs/models3b/parser_adapter; cp -r $A/runs/parser3b/best $A/runs/models3b/parser_adapter; }
  cd $A/runs && tar czf f3b.tgz f3b_*.log eval_parser_3b.json parser3b/train_log.json para_stats.json models3b 2>/dev/null; cd $A/src; echo "packed f3b $1 $(date)" >> $L; }
echo "f3b start $(date)" >> $L
timeout 2400 python paraphrase_data.py $M7 $D/parser_train.jsonl $D/para_train.jsonl 6000 1 > $A/runs/f3b_para.log 2>&1; echo "f3b_para rc=$? $(date)" >> $L
timeout 600 python paraphrase_data.py $M7 $D/parser_val.jsonl $D/para_val.jsonl 400 2 > $A/runs/f3b_para_val.log 2>&1; echo "f3b_para_val rc=$? $(date)" >> $L
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
echo "f3b_mix rc=$? $(date)" >> $L
timeout 3600 python train_parser.py train $D/mix_train.jsonl $D/mix_val.jsonl $A/runs/parser3b $M1 --epochs 2 --bs 8 --accum 2 > $A/runs/f3b_train.log 2>&1; echo "f3b_train rc=$? $(date)" >> $L; pack train
timeout 1500 python train_parser.py eval $A/runs/parser3b/best $D/parser_test.jsonl $A/runs/eval_parser_3b.json --base $M1 > $A/runs/f3b_eval.log 2>&1; echo "f3b_eval rc=$? $(date)" >> $L; pack eval
cp $D/para_train.jsonl $D/para_val.jsonl $A/runs/ 2>/dev/null; cd $A/runs && tar czf f3b_para.tgz para_train.jsonl para_val.jsonl 2>/dev/null; touch $A/F3B_DONE
