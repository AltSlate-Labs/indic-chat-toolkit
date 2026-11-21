#!/usr/bin/env bash
# lid-chat: romanized chat TRAINING data for indic-lid v3 from Ornith (vLLM, MIT; not the test teacher).
# 12 langs x 2000 requests, ~70 output tokens each (~1.7M tokens, ~420+ tok/s) -> ~60-70 min.
set -euo pipefail
cd ~/agent/rnd
O=jobs/lid-chat/out; mkdir -p $O
docker rm -f rnd-ornith 2>/dev/null || true
docker run -d --name rnd-ornith --network host --ipc host --gpus all --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $HOME/Downloads/model:/models:ro --entrypoint vllm vllm/vllm-openai:v0.32.0 serve /models/Ornith-1.5-35B-A3B-NVFP4 \
  --served-model-name ornith --host 127.0.0.1 --port 8011 --trust-remote-code --max-num-seqs 24 \
  --gpu-memory-utilization 0.45 --max-model-len 4096 --enable-prefix-caching --reasoning-parser qwen3
trap 'docker logs --tail 30 rnd-ornith > $HOME/agent/rnd/'$O'/vllm-tail.txt 2>&1 || true; docker rm -f rnd-ornith' EXIT
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8011/health >/dev/null && break; sleep 15; done
curl -sf http://127.0.0.1:8011/health >/dev/null || { echo "vllm not healthy"; exit 1; }
echo "[lid-chat] server up $(date -Is)"
. envs/lid/bin/activate
cd src/lid && python gen_chat_train.py http://127.0.0.1:8011 ornith ../../$O/chat-train.jsonl 2000
echo "[lid-chat] end $(date -Is)"
