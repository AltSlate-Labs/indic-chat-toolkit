#!/usr/bin/env bash
# tn-v3: tn-v2 misread or invented numbers when a sentence had more than one (every carrier has one slot): + 30k MULTI rows (2-3 filled
# carriers of one language joined), new held-out test-multi (1,000 rows, pairs of held-out carriers), sanity set (20 hand-written
# multi-number sentences, eyeballed only). Model = tn-v2 warm start, 1 epoch.
set -euo pipefail
cd ~/agent/rnd; O=jobs/tn-v3/out; G=jobs/tn-gap/out; mkdir -p $O
echo "[tn-v3] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
(cd src/tn && ../../envs/tn/bin/python fill_data.py ../../jobs/tn-lex/out/carriers.jsonl ../../$O 3)
envs/tn/bin/python - <<PY
import json
A={" ".join(json.loads(l)["text"].split()) for X in "AB" for l in open(f"$G/test{X}.jsonl",encoding="utf-8")}
for name in ("train","dev"):
    rows=[json.loads(l) for l in open(f"$O/{name}.jsonl",encoding="utf-8")]
    keep=[x for x in rows if " ".join(x["src"].split(": ",1)[1].split()) not in A]
    print(name, "dropped (exact match with tn-gap A/B):", len(rows)-len(keep))
    with open(f"$O/{name}.jsonl","w",encoding="utf-8") as f:
        for x in keep: f.write(json.dumps(x,ensure_ascii=False)+"\n")
rows=[json.loads(l) for l in open("$O/test-prog.jsonl",encoding="utf-8")]
with open("$O/test-prog-plain.jsonl","w",encoding="utf-8") as f:
    for x in rows: f.write(json.dumps({"lang":x["lang"],"text":x["src"].split(": ",1)[1]},ensure_ascii=False)+"\n")
PY
mkdir -p $O/base; envs/tn/bin/python src/tn/baselines.py $O/test-prog-plain.jsonl $O/base
. envs/pii/bin/activate
[ -s $O/model/train_meta.json ] || (cd src/xlit && python train_xlit.py --model ../../jobs/tn-v2/out/model --train ../../$O/train.jsonl \
   --dev ../../$O/dev.jsonl --out ../../$O/model --epochs 1 --bs 32 --lr 2e-4 --max_src 768 --max_tgt 1536)
python src/tn/predict_tn.py $O/model $O/test-prog.jsonl $O/pred-prog.jsonl
for M in tn-v2 tn-v3; do D=jobs/$M/out/model; python src/tn/predict_tn.py $D $O/test-multi.jsonl $O/pred-multi-$M.jsonl; done
python src/tn/predict_tn.py jobs/tn-v2/out/model $O/test-prog.jsonl $O/pred-prog-v2.jsonl
python src/tn/predict_tn.py $O/model src/tn/sanity.jsonl $O/pred-sanity.jsonl
for X in A B; do python src/tn/predict_tn.py $O/model $G/test$X.jsonl $O/pred-$X.jsonl; done
echo "== test-prog"
python src/tn/score_prog.py $O/test-prog.jsonl copy=$O/base/copy.jsonl rules=$O/base/rules.jsonl itp=$O/base/itp.jsonl \
   tn-v2=$O/pred-prog-v2.jsonl tn-v3=$O/pred-prog.jsonl
echo "== test-multi"
python src/tn/score_prog.py $O/test-multi.jsonl tn-v2=$O/pred-multi-tn-v2.jsonl tn-v3=$O/pred-multi-tn-v3.jsonl
paste -d'\t' src/tn/sanity.jsonl $O/pred-sanity.jsonl
deactivate
docker rm -f pii-teacher >/dev/null 2>&1 || true
docker run -d --name pii-teacher --gpus all -v $HOME/Downloads/model:/models -p 127.0.0.1:8080:8080 \
  ghcr.io/ggml-org/llama.cpp:server-cuda13 -m /models/Qwen3.8-27B-UD-Q4_K_XL-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -ngl 99 -c 32768 -np 8 --jinja --host 0.0.0.0 --port 8080 >/dev/null
trap 'docker rm -f pii-teacher >/dev/null 2>&1 || true' EXIT
for i in $(seq 1 120); do curl -sf 127.0.0.1:8080/health >/dev/null && break; sleep 5; done
for X in A B; do echo "== test $X"; envs/tn/bin/python src/tn/judge.py $G/test$X.jsonl tn-v3=$O/pred-$X.jsonl; done
echo "[tn-v3] end $(date -Is)"
