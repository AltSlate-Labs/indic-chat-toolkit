#!/usr/bin/env bash
# lid-testb: clean chat LID test (native-first, then self-romanized) from the TEST teacher, 12 langs x 60, ~20 min.
set -euo pipefail
cd ~/agent/rnd; O=jobs/lid-testb/out; mkdir -p $O
echo "[lid-testb] start $(date -Is)"
docker rm -f pii-teacher >/dev/null 2>&1 || true
docker run -d --name pii-teacher --gpus all -v $HOME/Downloads/model:/models -p 127.0.0.1:8080:8080 \
  ghcr.io/ggml-org/llama.cpp:server-cuda13 \
  -m /models/Qwen3.8-27B-UD-Q4_K_XL-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -ngl 99 -c 32768 -np 8 --jinja --host 0.0.0.0 --port 8080 >/dev/null
trap 'docker rm -f pii-teacher >/dev/null 2>&1 || true' EXIT
for i in $(seq 1 120); do curl -sf 127.0.0.1:8080/health >/dev/null && break; sleep 5; done
. envs/lid/bin/activate
python src/lid/make_chat_test.py $O/testb.jsonl 60
python src/lid/eval_lid.py $O/testb.jsonl
echo "[lid-testb] end $(date -Is)"
