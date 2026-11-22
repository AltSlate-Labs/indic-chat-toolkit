#!/usr/bin/env bash
# tn-v1: TTS normalizer. Data = Ornith carriers (jobs/tn-lex) filled by fill_data.py (hand lexicon + numtowords + CLDR) + Dakshina TRAIN wiki.
# Model = ByT5-small warm-started from xlit-v2 (model-fix), 2 epochs. Eval: test-prog exact match (held-out carriers) vs rules/copy;
# tn-gap A/B (never trained on) judged by the TEST teacher, compare with pass-79 baselines (rules A .607 / B .388, Qwen3-4B .068).
set -euo pipefail
cd ~/agent/rnd; O=jobs/tn-v1/out; G=jobs/tn-gap/out; mkdir -p $O
echo "[tn-v1] start $(date -Is)"
until grep -q "\[tn-lex\] end" jobs/tn-lex/log.txt; do sleep 30; done
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
(cd src/tn && ../../envs/tn/bin/python fill_data.py ../../jobs/tn-lex/out/carriers.jsonl ../../$O 3)
envs/tn/bin/python - <<PY
import json
rows=[json.loads(l) for l in open("$O/test-prog.jsonl",encoding="utf-8")]
with open("$O/test-prog-plain.jsonl","w",encoding="utf-8") as f:
    for x in rows: f.write(json.dumps({"lang":x["lang"],"text":x["src"].split(": ",1)[1]},ensure_ascii=False)+"\n")
PY
mkdir -p $O/base; envs/tn/bin/python src/tn/baselines.py $O/test-prog-plain.jsonl $O/base
. envs/pii/bin/activate
[ -s $O/model/train_meta.json ] || (cd src/xlit && python train_xlit.py --model ../../jobs/xlit-v2/out/model-fix --train ../../$O/train.jsonl \
   --dev ../../$O/dev.jsonl --out ../../$O/model --epochs 2 --bs 32 --lr 3e-4)
python src/tn/predict_tn.py $O/model $O/test-prog.jsonl $O/pred-prog.jsonl
for X in A B; do python src/tn/predict_tn.py $O/model $G/test$X.jsonl $O/pred-$X.jsonl; done
echo "== test-prog"
python src/tn/score_prog.py $O/test-prog.jsonl copy=$O/base/copy.jsonl rules=$O/base/rules.jsonl itp=$O/base/itp.jsonl tn-v1=$O/pred-prog.jsonl
deactivate
docker rm -f pii-teacher >/dev/null 2>&1 || true
docker run -d --name pii-teacher --gpus all -v $HOME/Downloads/model:/models -p 127.0.0.1:8080:8080 \
  ghcr.io/ggml-org/llama.cpp:server-cuda13 -m /models/Qwen3.8-27B-UD-Q4_K_XL-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -ngl 99 -c 32768 -np 8 --jinja --host 0.0.0.0 --port 8080 >/dev/null
trap 'docker rm -f pii-teacher >/dev/null 2>&1 || true' EXIT
for i in $(seq 1 120); do curl -sf 127.0.0.1:8080/health >/dev/null && break; sleep 5; done
for X in A B; do echo "== test $X"; envs/tn/bin/python src/tn/judge.py $G/test$X.jsonl tn-v1=$O/pred-$X.jsonl; done
echo "[tn-v1] end $(date -Is)"
