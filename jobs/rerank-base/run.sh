#!/usr/bin/env bash
# rerank-base: baselines on MIRACLReranking dev (hi, bn, te, en + romanized-query hi/bn/te), nDCG@10. ~15 min.
set -euo pipefail
cd ~/agent/rnd; M=~/Downloads/model; PY=envs/pii/bin/python
echo "[rerank-base] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
for r in BAAI/bge-reranker-v2-m3 cross-encoder/mmarco-mMiniLMv2-L12-H384-v1; do
  n=${r##*/}; [ -d $M/$n ] || $PY -c "from huggingface_hub import snapshot_download as s; s('$r', local_dir='$M/$n', allow_patterns=['*.json','*.safetensors','*.model','*.txt'])"
done
$PY src/rerank/eval_rerank.py first-stage-order=x:order bge-reranker-v2-m3-568M=$M/bge-reranker-v2-m3 \
  mmarco-mMiniLMv2-L12-118M=$M/mmarco-mMiniLMv2-L12-H384-v1 2>&1 | grep -v -i "warn\|Loading"
echo "[rerank-base] end $(date -Is)"
