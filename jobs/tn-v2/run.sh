#!/usr/bin/env bash
# tn-v2: fixes from the tn-v1 error analysis (test A judge .588 < rules .607):
#  no truncation (max_src 768 / max_tgt 1536; over-length rows dropped; tn-v1 trained on 7.7k inputs cut at 384 bytes),
#  wiki rows with years (per-language year style), integers up to 7 digits, glued suffixes and ranges (6k/lang), SCORE 10 fills/carrier.
# Model = tn-v1 warm start, 1 epoch. Same held-out carriers as v1 (test-prog gold changes only where the year style changed).
set -euo pipefail
cd ~/agent/rnd; O=jobs/tn-v2/out; G=jobs/tn-gap/out; mkdir -p $O
echo "[tn-v2] start $(date -Is)"
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
[ -s $O/model/train_meta.json ] || (cd src/xlit && python train_xlit.py --model ../../jobs/tn-v1/out/model --train ../../$O/train.jsonl \
   --dev ../../$O/dev.jsonl --out ../../$O/model --epochs 1 --bs 32 --lr 2e-4 --max_src 768 --max_tgt 1536)
python src/tn/predict_tn.py $O/model $O/test-prog.jsonl $O/pred-prog.jsonl
python src/tn/predict_tn.py jobs/tn-v1/out/model $O/test-prog.jsonl $O/pred-prog-v1.jsonl
for X in A B; do python src/tn/predict_tn.py $O/model $G/test$X.jsonl $O/pred-$X.jsonl; done
echo "== test-prog"
python src/tn/score_prog.py $O/test-prog.jsonl copy=$O/base/copy.jsonl rules=$O/base/rules.jsonl itp=$O/base/itp.jsonl \
   tn-v1=$O/pred-prog-v1.jsonl tn-v2=$O/pred-prog.jsonl
deactivate
docker rm -f pii-teacher >/dev/null 2>&1 || true
docker run -d --name pii-teacher --gpus all -v $HOME/Downloads/model:/models -p 127.0.0.1:8080:8080 \
  ghcr.io/ggml-org/llama.cpp:server-cuda13 -m /models/Qwen3.8-27B-UD-Q4_K_XL-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -ngl 99 -c 32768 -np 8 --jinja --host 0.0.0.0 --port 8080 >/dev/null
trap 'docker rm -f pii-teacher >/dev/null 2>&1 || true' EXIT
for i in $(seq 1 120); do curl -sf 127.0.0.1:8080/health >/dev/null && break; sleep 5; done
for X in A B; do echo "== test $X"; envs/tn/bin/python src/tn/judge.py $G/test$X.jsonl tn-v2=$O/pred-$X.jsonl; done
echo "[tn-v2] end $(date -Is)"
