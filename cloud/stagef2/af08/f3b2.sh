#!/usr/bin/env bash
# F3b after the flights (their 20 perception copies filled the GPU; the 7B needs ~15 GB). Same steps as setup_f2.sh part (1).
set -u; A=$HOME/af08; L=$A/runs/setup.log; D=$A/data; M1=$HOME/models/qwen1.5b; M7=$HOME/models/qwen7b
export OSM2_DIR=$A/osm2 HF_HUB_DISABLE_TELEMETRY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; . $HOME/venv/bin/activate; cd $A/src
step() { echo "$1 rc=$2 $(date)" >> $L; [ "$2" != "0" ] && { echo "$1 FAILED" >> $L; tail -20 $A/runs/$1.log >> $L 2>/dev/null; }; return 0; }
pack() { cd $A/runs && tar czf f3b.tgz f3b_*.log eval_parser_3b.json parser3b_train_log.json para_stats.json para_train.jsonl para_val.jsonl models3b 2>/dev/null; cd $A/src; echo "packed f3b $1 $(date)" >> $L; }
while pgrep -f 'test_[a]ctions.py' > /dev/null; do sleep 30; done; echo "f3b2 start $(date)" >> $L
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
mkdir -p $A/runs/models3b; [ -d $A/runs/parser3b/best ] && { rm -rf $A/runs/models3b/parser_adapter; cp -r $A/runs/parser3b/best $A/runs/models3b/parser_adapter; }; cp $A/runs/parser3b/train_log.json $A/runs/parser3b_train_log.json 2>/dev/null; pack train
timeout 1800 python train_parser.py eval $A/runs/parser3b/best $D/parser_test.jsonl $A/runs/eval_parser_3b.json --base $M1 > $A/runs/f3b_eval.log 2>&1; step f3b_eval $?; pack eval
touch $A/F3B_DONE
