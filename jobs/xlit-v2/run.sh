#!/usr/bin/env bash
# xlit-v2: warm-start from xlit-v1 (model-fix) on v1 data + Aksharantar words + long pseudo-sentences (see build_data_v2.py).
# Ship only if A and B CER <= v1 (.063/.117), kn/ml/te improve, and loops (pred > 1.3x ref + 5) drop.
set -euo pipefail
cd ~/agent/rnd/src/xlit && . ../../envs/pii/bin/activate
O=../../jobs/xlit-v2/out; T=../../jobs/xlit-gap/out; mkdir -p $O
echo "[xlit-v2] start $(date -Is)"
[ -s $O/train.jsonl ] || python build_data_v2.py ../../jobs/xlit-v1/out $O
[ -s $O/model/train_meta.json ] || python train_xlit.py --model ../../jobs/xlit-v1/out/model-fix --train $O/train.jsonl --dev $O/dev.jsonl --out $O/model --epochs 1 --bs 32 --lr 2e-4
for X in A B; do python predict.py $O/model $T/test$X.jsonl $O/pred-$X.jsonl; echo "== xlit-v2 test $X"; python metrics.py $T/test$X.jsonl $O/pred-$X.jsonl; done
python3 - <<PY
import json
for X in "AB":
    t=[json.loads(l) for l in open(f"$T/test{X}.jsonl")]
    for v in ("v1","v2"):
        p=[json.loads(l)["pred"] for l in open(f"../../jobs/xlit-{v}/out/pred-{X}.jsonl")]
        print(X, v, "loops", [x["lang"] for x,q in zip(t,p) if len(q)>1.3*len(x["native"])+5])
PY
echo "[xlit-v2] end $(date -Is)"
