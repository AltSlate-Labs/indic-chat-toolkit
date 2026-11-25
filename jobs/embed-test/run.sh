#!/usr/bin/env bash
# One-time TEST-split evaluation of the chosen student (embed-v3, picked on dev) and the 5 baselines. ~15 min.
set -euo pipefail
cd ~/agent/rnd; M=~/Downloads/model
echo "[embed-test] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
envs/pii/bin/python src/embed/evaluate.py test \
  embed-v3=jobs/embed-v3/out:mean_head \
  e5-small-118M=$M/multilingual-e5-small:mean:"query: " \
  MiniLM-L12-118M=$M/paraphrase-multilingual-MiniLM-L12-v2:mean \
  l3cube-indic-sbert-237M=$M/indic-sentence-similarity-sbert:mean \
  LaBSE-471M=$M/LaBSE:cls_dense \
  bge-m3-568M=$M/bge-m3:cls 2>&1 | grep -v -i "warn\|Loading\|deprecated"
echo "[embed-test] end $(date -Is)"
