#!/usr/bin/env bash
# tn-gap phase 1b: baselines (copy, rules, indic-tts-preprocess, Qwen3-4B-Instruct prompted) on tests A and B, judged by the TEST teacher.
set -euo pipefail
cd ~/agent/rnd; O=jobs/tn-gap/out
echo "[tn-gap2] start $(date -Is)"
until grep -q "\[tn-gap\] end" jobs/tn-gap/log.txt; do sleep 30; done
for X in A B; do mkdir -p $O/pred$X; envs/tn/bin/python src/tn/baselines.py $O/test$X.jsonl $O/pred$X; done
docker rm -f rnd-small >/dev/null 2>&1 || true
docker run -d --name rnd-small --gpus all -v $HOME/Downloads/model:/models -p 127.0.0.1:8012:8000 vllm/vllm-openai:v0.32.0 \
  --model /models/Qwen3-4B-Instruct-2507 --served-model-name small --max-model-len 4096 --gpu-memory-utilization 0.35 >/dev/null
trap 'docker rm -f rnd-small pii-teacher >/dev/null 2>&1 || true' EXIT
for i in $(seq 1 120); do curl -sf 127.0.0.1:8012/health >/dev/null && break; sleep 5; done
for X in A B; do envs/tn/bin/python src/tn/llm_norm.py $O/test$X.jsonl $O/pred$X/qwen3-4b.jsonl http://127.0.0.1:8012/v1/chat/completions small; done
docker rm -f rnd-small >/dev/null
docker run -d --name pii-teacher --gpus all -v $HOME/Downloads/model:/models -p 127.0.0.1:8080:8080 \
  ghcr.io/ggml-org/llama.cpp:server-cuda13 -m /models/Qwen3.8-27B-UD-Q4_K_XL-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -ngl 99 -c 32768 -np 8 --jinja --host 0.0.0.0 --port 8080 >/dev/null
for i in $(seq 1 120); do curl -sf 127.0.0.1:8080/health >/dev/null && break; sleep 5; done
for X in A B; do echo "== test $X"; envs/tn/bin/python src/tn/judge.py $O/test$X.jsonl copy=$O/pred$X/copy.jsonl rules=$O/pred$X/rules.jsonl itp=$O/pred$X/itp.jsonl qwen3-4b=$O/pred$X/qwen3-4b.jsonl; done
echo "[tn-gap2] end $(date -Is)"
