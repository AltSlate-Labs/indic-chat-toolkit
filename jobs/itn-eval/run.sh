#!/usr/bin/env bash
# itn-eval: real-sentence ITN tests (made in pass 108 after a CPU preview showed test-B's teacher spoken forms often read a
# different number than the reference, e.g. ur 1985 -> "ننانوے سو پچپن", ta ₹1,50,000 -> garbled):
#  test-A-itn  : 568 Dakshina VALID sentences, spoken by our rules (correct by construction), ref = original (src/itn/make_tests_real.py)
#  test-B-clean: test B rows whose teacher spoken form the judge (Qwen3.8-27B) accepts as a correct reading of the original.
# usage: bash jobs/itn-eval/run.sh itn-v1
set -euo pipefail
M=$1; cd ~/agent/rnd; O=jobs/itn-eval/out; P=jobs/$M/out; B=jobs/itn-data/out/base; mkdir -p $O/$M
echo "[itn-eval $M] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
. envs/pii/bin/activate
python src/tn/predict_tn.py $P/model $O/test-A-itn.jsonl $O/$M/pred-A.jsonl
deactivate
if [ ! -s $O/testB-teacher.jsonl.judge ]; then
  docker rm -f pii-teacher >/dev/null 2>&1 || true
  docker run -d --name pii-teacher --gpus all -v $HOME/Downloads/model:/models -p 127.0.0.1:8080:8080 \
    ghcr.io/ggml-org/llama.cpp:server-cuda13 -m /models/Qwen3.8-27B-UD-Q4_K_XL-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf \
    -ngl 99 -c 32768 -np 8 --jinja --host 0.0.0.0 --port 8080 >/dev/null
  trap 'docker rm -f pii-teacher >/dev/null 2>&1 || true' EXIT
  for i in $(seq 1 120); do curl -sf 127.0.0.1:8080/health >/dev/null && break; sleep 5; done
  echo "== teacher spoken forms of test B, judged"
  envs/tn/bin/python src/tn/judge.py jobs/tn-gap/out/testB.jsonl teacher=$O/testB-teacher.jsonl
  docker rm -f pii-teacher >/dev/null 2>&1 || true
fi
python3 - <<PY
import json
ok=json.load(open("$O/testB-teacher.jsonl.judge"))
def sub(src, dst):
    rows=[l for l, k in zip(open(src, encoding="utf-8"), ok) if k]
    open(dst, "w", encoding="utf-8").writelines(rows); return len(rows)
print("test-B-clean rows:", sub("$P/test-B.jsonl", "$O/test-B-clean.jsonl"))
for name, f in (("copy", "$B/copy-B.jsonl"), ("ip", "$B/ip-B.jsonl"), ("$M", "$P/pred-B.jsonl")):
    sub(f, f"$O/$M/clean-{name}.jsonl")
PY
echo "== test-A-itn (Dakshina valid, spoken by rules)"
python3 src/itn/score_itn.py $O/test-A-itn.jsonl copy=$O/base/copy-A.jsonl indic-punct=$O/base/ip-A.jsonl $M=$O/$M/pred-A.jsonl
echo "== test-B-clean (teacher spoken forms the judge accepts)"
python3 src/itn/score_itn.py $O/test-B-clean.jsonl copy=$O/$M/clean-copy.jsonl indic-punct=$O/$M/clean-ip.jsonl $M=$O/$M/clean-$M.jsonl
echo "[itn-eval $M] end $(date -Is)"
