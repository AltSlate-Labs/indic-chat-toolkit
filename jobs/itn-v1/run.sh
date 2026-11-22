#!/usr/bin/env bash
# itn-v1: inverse text normalization (spoken -> written) for 10 Indian languages; plan in notes/itn-plan.md.
# Data = the TN generator reversed (src/tn/fill_data.py "itn" field -> src/itn/make_itn.py), canonical written targets.
# Model = tn-v4b warm start (knows the number words), prefix "<lang>-itn:", 1 epoch (~5.1k steps x 1.65 s = 2.3 h, checkpoints).
# Tests: test-prog/multi/pair (held-out carriers, exact + digits), test-B = tn-gap test B reversed (digits); baselines copy +
# indic-punct were run on CPU in pass 107 (jobs/itn-data/out/base).
set -euo pipefail
cd ~/agent/rnd; O=jobs/itn-v1/out; T=$O/tn; G=jobs/tn-gap/out; mkdir -p $T
echo "[itn-v1] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
(cd src/tn && ../../envs/tn/bin/python fill_data.py ../../jobs/tn-lex/out/carriers.jsonl ../../$T 3 >/dev/null)
python3 src/itn/make_itn.py $T $G/testB.jsonl $O
python3 - <<PY
import json
# drop train/dev rows whose spoken input or written target equals a test-B (or test-A) sentence
B={" ".join(json.loads(l)[k].split()) for X in "AB" for l in open(f"$G/test{X}.jsonl",encoding="utf-8") for k in ("text","spoken") if k in json.loads(l)}
for name in ("train","dev"):
    rows=[json.loads(l) for l in open(f"$O/{name}.jsonl",encoding="utf-8")]
    keep=[x for x in rows if " ".join(x["src"].split(": ",1)[1].split()) not in B and " ".join(x["tgt"].split()) not in B]
    print(name,"dropped (match with tn-gap A/B):",len(rows)-len(keep))
    with open(f"$O/{name}.jsonl","w",encoding="utf-8") as f:
        for x in keep: f.write(json.dumps(x,ensure_ascii=False)+"\n")
PY
. envs/pii/bin/activate
[ -s $O/model/train_meta.json ] || (cd src/xlit && GPU_MEM_FRAC=0.6 python train_xlit.py --model ../../jobs/tn-v4b/out/model --train ../../$O/train.jsonl \
   --dev ../../$O/dev.jsonl --out ../../$O/model --epochs 1 --bs 32 --lr 2e-4 --max_src 768 --max_tgt 1536)
for X in prog multi pair B; do python src/tn/predict_tn.py $O/model $O/test-$X.jsonl $O/pred-$X.jsonl; done
B=jobs/itn-data/out/base
echo "== test-prog"; python3 src/itn/score_itn.py $O/test-prog.jsonl copy=$B/copy-prog.jsonl indic-punct=$B/ip-prog.jsonl itn-v1=$O/pred-prog.jsonl
echo "== test-multi"; python3 src/itn/score_itn.py $O/test-multi.jsonl itn-v1=$O/pred-multi.jsonl
echo "== test-pair"; python3 src/itn/score_itn.py $O/test-pair.jsonl itn-v1=$O/pred-pair.jsonl
echo "== test-B"; python3 src/itn/score_itn.py $O/test-B.jsonl copy=$B/copy-B.jsonl indic-punct=$B/ip-B.jsonl itn-v1=$O/pred-B.jsonl
echo "[itn-v1] end $(date -Is)"
