#!/bin/bash
# Qwen3.8-27B judge (test judge) on the same 300 PHINC dev rows (seed 3) for mt-v3b (mt-v3 3.50, mt-v2 3.25, Ornith 4.35, human refs 3.93).
cd ~/agent/rnd; PY=envs/pii/bin/python; O=jobs/mt-eval6/out; B=jobs/mt-base/out
echo "[mt-eval6] start $(date -Is)"
while tmux has -t job-mt-v3b 2>/dev/null; do sleep 30; done
[ -s $B/pred-mt-v3b-phinc.jsonl ] || { echo "no mt-v3 preds; stop"; exit 1; }
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
mkdir -p $O
docker rm -f pii-teacher >/dev/null 2>&1 || true
docker run -d --name pii-teacher --gpus all -v $HOME/Downloads/model:/models -p 127.0.0.1:8080:8080 \
  ghcr.io/ggml-org/llama.cpp:server-cuda13 -m /models/Qwen3.8-27B-UD-Q4_K_XL-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -ngl 99 -c 32768 -np 8 --jinja --host 0.0.0.0 --port 8080 >/dev/null
trap 'docker rm -f pii-teacher >/dev/null 2>&1 || true' EXIT
for i in $(seq 1 120); do curl -sf 127.0.0.1:8080/health >/dev/null && break; sleep 5; done
$PY src/mt/judge_mt.py 300 $O/judge-phinc-dev.json mt-v3b=$B/pred-mt-v3b-phinc.jsonl 2>&1 | grep "judge\|Error"
echo "[mt-eval6] end $(date -Is)"
