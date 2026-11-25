#!/usr/bin/env bash
# rerank-data: MIRACL train candidate lists (judged + top-30 indic-embed-small hard negatives) scored by bge-reranker-v2-m3.
# Corpus embedding ~1.5M passages + ~250k teacher pairs; estimate ~40 min.
set -euo pipefail
cd ~/agent/rnd; M=~/Downloads/model; PY=envs/pii/bin/python
echo "[rerank-data] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
$PY src/rerank/build_lists.py jobs/rerank-data/out jobs/embed-v3/out $M/bge-reranker-v2-m3 30 2>&1 | grep --line-buffered -v -i "warn\|Loading"
echo "[rerank-data] end $(date -Is)"
