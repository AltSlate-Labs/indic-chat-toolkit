#!/usr/bin/env bash
# tn-v5-audit: why does the judge reject tn-v5 outputs? Judge names the error for up to 15 NOs per language on tests A and B.
set -euo pipefail
cd ~/agent/rnd; O=jobs/tn-v5-audit/out; mkdir -p $O; P=jobs/nfc-v5/out; G=jobs/tn-gap/out
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
docker rm -f pii-teacher >/dev/null 2>&1 || true
docker run -d --name pii-teacher --gpus all -v $HOME/Downloads/model:/models -p 127.0.0.1:8080:8080 \
  ghcr.io/ggml-org/llama.cpp:server-cuda13 -m /models/Qwen3.8-27B-UD-Q4_K_XL-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -ngl 99 -c 32768 -np 8 --jinja --host 0.0.0.0 --port 8080 >/dev/null
trap 'docker rm -f pii-teacher >/dev/null 2>&1 || true' EXIT
for i in $(seq 1 120); do curl -sf 127.0.0.1:8080/health >/dev/null && break; sleep 5; done
echo "[tn-v5-audit] start $(date -Is)"
for X in A B; do for l in hi bn ta te mr gu kn ml pa ur; do
  python3 src/tn/judge_why.py $G/test$X.jsonl $P/tn-pred-$X.jsonl $P/tn-pred-$X.jsonl.judge $l 15 $O/why-$X-$l.jsonl; done; done
echo "[tn-v5-audit] end $(date -Is)"
