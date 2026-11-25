#!/usr/bin/env bash
# rerank-v1: mmBERT-small cross-encoder distilled from bge-reranker-v2-m3 on MIRACL train lists (native + romanized queries).
# 6000 steps x 16 queries x 16 passages; step time measured in the first 100 steps (estimate ~0.6 s -> ~1 h). Then MIRACLReranking dev.
set -euo pipefail
cd ~/agent/rnd; M=~/Downloads/model; PY=envs/pii/bin/python; O=jobs/rerank-v1/out
echo "[rerank-v1] start $(date -Is)"
while tmux has-session -t job-rerank-data 2>/dev/null; do sleep 30; done
grep -q "^\[rerank-data\] end" jobs/rerank-data/log.txt || { echo "rerank-data did not finish"; exit 1; }
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
[ -s $O/train_meta.json ] || $PY src/rerank/train_rerank.py --lists jobs/rerank-data/out/lists.jsonl --base $M/mmBERT-small --out $O \
  --steps 6000 --q 16 --g 16 --lr 3e-5 2>&1 | grep --line-buffered -v -i "warn\|Loading"
echo "== MIRACLReranking dev nDCG@10 (teacher native .857 roman .560; mMiniLM .774/.264; no-rerank .565)"
$PY src/rerank/eval_rerank.py rerank-v1=$O 2>&1 | grep native
echo "[rerank-v1] end $(date -Is)"
