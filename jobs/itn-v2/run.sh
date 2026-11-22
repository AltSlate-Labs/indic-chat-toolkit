#!/usr/bin/env bash
# itn-v2: itn-v1 drops the zero after "thousand" (probe: 1024 -> 124 ta/ml, 4096 -> 496 te/ta/ml/bn; test-A ml 1077 -> 177).
# + 15k rows with internal-zero numbers (src/itn/zeros.py, training carriers only) mixed with 45k itn-v1 rows; warm start itn-v1,
# 1 epoch of the 60k mix (~1.9k steps x 2.1 s = 65 min). New held-out test: zeros-test (300 rows, held-out carriers).
set -euo pipefail
cd ~/agent/rnd; O=jobs/itn-v2/out; P=jobs/itn-v1/out; E=jobs/itn-eval/out; Z=jobs/itn-eval/zeros; mkdir -p $O
echo "[itn-v2] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
python3 - <<PY
import json, random
r=random.Random(0); a=[l for l in open("$P/train.jsonl",encoding="utf-8")]; z=[l for l in open("$Z/zeros-train.jsonl",encoding="utf-8")]
rows=r.sample(a,45000)+z; r.shuffle(rows); open("$O/train.jsonl","w",encoding="utf-8").writelines(rows); print("train", len(rows))
PY
. envs/pii/bin/activate
[ -s $O/model/train_meta.json ] || (cd src/xlit && GPU_MEM_FRAC=0.6 python train_xlit.py --model ../../$P/model --train ../../$O/train.jsonl \
   --dev ../../$P/dev.jsonl --out ../../$O/model --epochs 1 --bs 32 --lr 1e-4 --max_src 768 --max_tgt 1536)
python src/tn/predict_tn.py $P/model $Z/zeros-test.jsonl $O/pred-zeros-v1.jsonl
for X in prog multi pair B; do python src/tn/predict_tn.py $O/model $P/test-$X.jsonl $O/pred-$X.jsonl; done
python src/tn/predict_tn.py $O/model $Z/zeros-test.jsonl $O/pred-zeros.jsonl
python src/tn/predict_tn.py $O/model $E/test-A-itn.jsonl $O/pred-A.jsonl
B=jobs/itn-data/out/base
echo "== zeros-test"; python3 src/itn/score_itn.py $Z/zeros-test.jsonl itn-v1=$O/pred-zeros-v1.jsonl itn-v2=$O/pred-zeros.jsonl
echo "== test-prog"; python3 src/itn/score_itn.py $P/test-prog.jsonl indic-punct=$B/ip-prog.jsonl itn-v1=$P/pred-prog.jsonl itn-v2=$O/pred-prog.jsonl
echo "== test-multi"; python3 src/itn/score_itn.py $P/test-multi.jsonl itn-v1=$P/pred-multi.jsonl itn-v2=$O/pred-multi.jsonl
echo "== test-pair"; python3 src/itn/score_itn.py $P/test-pair.jsonl itn-v1=$P/pred-pair.jsonl itn-v2=$O/pred-pair.jsonl
echo "== test-A-itn"; python3 src/itn/score_itn.py $E/test-A-itn.jsonl indic-punct=$E/base/ip-A.jsonl itn-v1=$E/itn-v1/pred-A.jsonl itn-v2=$O/pred-A.jsonl
echo "== test-B (raw, noisy)"; python3 src/itn/score_itn.py $P/test-B.jsonl indic-punct=$B/ip-B.jsonl itn-v1=$P/pred-B.jsonl itn-v2=$O/pred-B.jsonl
cp $P/test-B.jsonl $O/test-B.jsonl  # so jobs/itn-eval/run.sh itn-v2 finds it
echo "[itn-v2] end $(date -Is)"
