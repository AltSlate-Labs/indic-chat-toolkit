#!/usr/bin/env bash
# tn-lex: per-language lexicon (CLDR + Ornith x3) + carrier sentences (gen_carriers, 30 calls/lang/kind) for the TTS normalizer.
set -euo pipefail
cd ~/agent/rnd; O=jobs/tn-lex; mkdir -p $O/out
echo "[tn-lex] start $(date -Is)"
docker rm -f rnd-ornith >/dev/null 2>&1 || true
docker run -d --name rnd-ornith --network host --ipc host --gpus all --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $HOME/Downloads/model:/models:ro --entrypoint vllm vllm/vllm-openai:v0.32.0 serve /models/Ornith-1.5-35B-A3B-NVFP4 \
  --served-model-name ornith --host 127.0.0.1 --port 8011 --trust-remote-code --max-num-seqs 24 \
  --gpu-memory-utilization 0.45 --max-model-len 4096 --enable-prefix-caching --reasoning-parser qwen3 >/dev/null
trap 'docker logs --tail 30 rnd-ornith > $HOME/agent/rnd/'$O'/vllm-tail.txt 2>&1 || true; docker rm -f rnd-ornith' EXIT
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8011/health >/dev/null && break; sleep 15; done
curl -sf http://127.0.0.1:8011/health >/dev/null || { echo "vllm not healthy"; exit 1; }
envs/tn/bin/python src/tn/make_lexicon.py $O/out http://127.0.0.1:8011
echo "[tn-lex] carriers start $(date -Is)"
[ -s $O/out/carriers.jsonl ] || envs/tn/bin/python src/tn/gen_carriers.py http://127.0.0.1:8011 $O/out/carriers.jsonl 30
echo "[tn-lex] end $(date -Is)"
