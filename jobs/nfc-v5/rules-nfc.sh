#!/usr/bin/env bash
# Fair baseline for the TN card: rules (indic-numtowords) outputs have the same decomposed Telugu ై; judge them NFC-normalized.
# Also tn-v5 on test-multi / test-pair (NFC).
set -euo pipefail
cd ~/agent/rnd; O=jobs/nfc-v5/out; G=jobs/tn-gap/out; T=jobs/tn-v4b/out
nfc() { python3 -c "
import json,sys,unicodedata as u
for l in open(sys.argv[1],encoding='utf-8'):
    x=json.loads(l); print(json.dumps({k:(u.normalize('NFC',v) if isinstance(v,str) else v) for k,v in x.items()},ensure_ascii=False))" $1 > $2; }
for X in A B; do nfc $G/pred$X/rules.jsonl $O/rules-$X-nfc.jsonl; done
nfc $T/test-pair.jsonl $O/tn-test-pair.jsonl
. envs/pii/bin/activate
[ -s $O/tn-pred-pair.jsonl ] || python src/tn/predict_tn.py $O/tn-v5 $O/tn-test-pair.jsonl $O/tn-pred-pair.jsonl
[ -s jobs/tn-v5/out/pred-multi-tn-v5.jsonl ] || true
deactivate
echo "== tn-v5 test-multi / test-pair"
envs/tn/bin/python src/tn/score_prog.py jobs/tn-v5/out/test-multi.jsonl tn-v5=jobs/tn-v5/out/pred-multi-tn-v5.jsonl | head -1
envs/tn/bin/python src/tn/score_prog.py $O/tn-test-pair.jsonl tn-v5=$O/tn-pred-pair.jsonl | head -1
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
docker rm -f pii-teacher >/dev/null 2>&1 || true
docker run -d --name pii-teacher --gpus all -v $HOME/Downloads/model:/models -p 127.0.0.1:8080:8080 \
  ghcr.io/ggml-org/llama.cpp:server-cuda13 -m /models/Qwen3.8-27B-UD-Q4_K_XL-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -ngl 99 -c 32768 -np 8 --jinja --host 0.0.0.0 --port 8080 >/dev/null
trap 'docker rm -f pii-teacher >/dev/null 2>&1 || true' EXIT
for i in $(seq 1 120); do curl -sf 127.0.0.1:8080/health >/dev/null && break; sleep 5; done
for X in A B; do echo "== rules test $X, NFC (raw: A .607 / B .388)"; envs/tn/bin/python src/tn/judge.py $G/test$X.jsonl rules-nfc=$O/rules-$X-nfc.jsonl; done
echo "[rules-nfc] end $(date -Is)"
