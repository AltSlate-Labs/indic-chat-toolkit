#!/usr/bin/env bash
# tn-judge-audit: is TN's low Telugu judge score (test A .32) real errors or judge bias? Ask the same judge to name the error for
# 40 te NO verdicts, plus 20 each for ta and hi as controls.
set -euo pipefail
cd ~/agent/rnd; O=jobs/tn-judge-audit/out; mkdir -p $O; P=jobs/tn-v4b/out
docker rm -f pii-teacher >/dev/null 2>&1 || true
docker run -d --name pii-teacher --gpus all -v $HOME/Downloads/model:/models -p 127.0.0.1:8080:8080 \
  ghcr.io/ggml-org/llama.cpp:server-cuda13 -m /models/Qwen3.8-27B-UD-Q4_K_XL-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -ngl 99 -c 32768 -np 8 --jinja --host 0.0.0.0 --port 8080 >/dev/null
trap 'docker rm -f pii-teacher >/dev/null 2>&1 || true' EXIT
for i in $(seq 1 120); do curl -sf 127.0.0.1:8080/health >/dev/null && break; sleep 5; done
echo "[tn-judge-audit] start $(date -Is)"
for l in "te 40" "ta 20" "hi 20"; do set -- $l
  python3 src/tn/judge_why.py jobs/tn-gap/out/testA.jsonl $P/pred-A.jsonl $P/pred-A.jsonl.judge $1 $2 $O/why-$1.jsonl; done
echo "[tn-judge-audit] end $(date -Is)"
