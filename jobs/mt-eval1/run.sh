#!/usr/bin/env bash
# After mt-v1: FLORES dev (100/lang, native + our-romanizer) for copy / opus-mt-mul-en / mt-v1; then Qwen3.8-27B (test judge;
# upper bound, not a training teacher) on PHINC dev and FLORES dev. All DEV splits.
set -euo pipefail
cd ~/agent/rnd; PY=envs/pii/bin/python; M=~/Downloads/model; F=jobs/mt-eval1/out/flores-dev.jsonl; P=jobs/mt-base/out/dev.jsonl
echo "[mt-eval1] start $(date -Is)"
while tmux has -t job-mt-v1 2>/dev/null; do sleep 30; done
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
mkdir -p jobs/mt-eval1/out; [ -s $F ] || $PY src/mt/make_flores_mt.py dev $F 100
for spec in "copy copy x" "opus-mt-mul-en marian $M/opus-mt-mul-en" "mt-v1 marian jobs/mt-v1/out/model"; do
  set -- $spec; $PY src/mt/eval_mt.py $F $1-flores $2 $3 2>&1 | grep "chrF"
done
docker rm -f pii-teacher >/dev/null 2>&1 || true
docker run -d --name pii-teacher --gpus all -v $HOME/Downloads/model:/models -p 127.0.0.1:8080:8080 \
  ghcr.io/ggml-org/llama.cpp:server-cuda13 -m /models/Qwen3.8-27B-UD-Q4_K_XL-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -ngl 99 -c 32768 -np 8 --jinja --host 0.0.0.0 --port 8080 >/dev/null
trap 'docker rm -f pii-teacher >/dev/null 2>&1 || true' EXIT
for i in $(seq 1 120); do curl -sf 127.0.0.1:8080/health >/dev/null && break; sleep 5; done
$PY src/mt/eval_mt.py $P qwen27b-phinc llm x 2>&1 | grep "chrF"
$PY src/mt/eval_mt.py $F qwen27b-flores llm x 2>&1 | grep "chrF"
echo "[mt-eval1] end $(date -Is)"
