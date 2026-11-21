#!/usr/bin/env bash
# xlit-v1: warm-start from xlit-v0 (model-fix) on v1 data (more lexicon words, lexicon-covered wiki sentences, keep-English "+en" mode).
set -euo pipefail
cd ~/agent/rnd/src/xlit && . ../../envs/pii/bin/activate
O=../../jobs/xlit-v1/out; T=../../jobs/xlit-gap/out; mkdir -p $O
echo "[xlit-v1] start $(date -Is)"
python build_data_v1.py $O
[ -s $O/model/train_meta.json ] || python train_xlit.py --model ../../jobs/xlit-v0/out/model-fix --train $O/train.jsonl --dev $O/dev.jsonl --out $O/model --epochs 1 --bs 32 --lr 2e-4
for X in A B; do python predict.py $O/model $T/test$X.jsonl $O/pred-$X.jsonl; echo "== xlit-v1 test $X"; python metrics.py $T/test$X.jsonl $O/pred-$X.jsonl; done
python predict.py $O/model $T/testB.jsonl $O/pred-B-en.jsonl 1 +en
python - <<PY
import json
t=[json.loads(l) for l in open("$T/testB.jsonl")]; a=[json.loads(l)["pred"] for l in open("$O/pred-B.jsonl")]; b=[json.loads(l)["pred"] for l in open("$O/pred-B-en.jsonl")]
for i in range(0, len(t), 58): print(t[i]["lang"], "|", t[i]["roman"], "\n   default:", a[i], "\n   +en:    ", b[i])
PY
echo "[xlit-v1] end $(date -Is)"
