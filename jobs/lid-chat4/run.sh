#!/usr/bin/env bash
# lid-chat4: v4 TRAINING chat: 9 out-of-scope Indic langs (-> other) x1500 + weak ur,pa,ta,ml,kn x1500 (new seeds) from Ornith (vLLM, MIT; not the test teacher).
set -euo pipefail
cd ~/agent/rnd
O=jobs/lid-chat4/out; mkdir -p $O
docker rm -f rnd-ornith 2>/dev/null || true
docker run -d --name rnd-ornith --network host --ipc host --gpus all --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $HOME/Downloads/model:/models:ro --entrypoint vllm vllm/vllm-openai:v0.32.0 serve /models/Ornith-1.5-35B-A3B-NVFP4 \
  --served-model-name ornith --host 127.0.0.1 --port 8011 --trust-remote-code --max-num-seqs 24 \
  --gpu-memory-utilization 0.45 --max-model-len 4096 --enable-prefix-caching --reasoning-parser qwen3
trap 'docker logs --tail 30 rnd-ornith > $HOME/agent/rnd/'$O'/vllm-tail.txt 2>&1 || true; docker rm -f rnd-ornith' EXIT
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8011/health >/dev/null && break; sleep 15; done
curl -sf http://127.0.0.1:8011/health >/dev/null || { echo "vllm not healthy"; exit 1; }
echo "[lid-chat4] server up $(date -Is)"
. envs/lid/bin/activate
cd src/lid
GEN_LANGS=as,or,ne,mai,sa,kok,ks,mni,brx python gen_chat_train.py http://127.0.0.1:8011 ornith ../../$O/chat-oos.jsonl 1500
GEN_LANGS=ur,pa,ta,ml,kn SEED_OFF=500000 python gen_chat_train.py http://127.0.0.1:8011 ornith ../../$O/chat-weak.jsonl 1500
echo "[lid-chat4] end $(date -Is)"
