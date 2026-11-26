#!/usr/bin/env bash
# rerank-v2: add English. NQ pairs (sentence-transformers/natural-questions, from Natural Questions CC BY-SA 3.0) -> 6000 English
# lists (gold + top-30 indic-embed-small negatives, bge-reranker-v2-m3 scores); continue rerank-v1 for 3000 steps (lr 2e-5) on
# v1 lists + English lists; MIRACLReranking dev eval. ~15 min data + 3000 x 2.2 s = ~1.8 h + 25 min eval.
set -euo pipefail
cd ~/agent/rnd; M=~/Downloads/model; PY=envs/pii/bin/python; O=jobs/rerank-v2/out; NQ=~/Downloads/data/st-natural-questions
echo "[rerank-v2] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
[ -s $NQ/pair/train-00000-of-00001.parquet ] || $PY -c "from huggingface_hub import snapshot_download as s; s('sentence-transformers/natural-questions', repo_type='dataset', local_dir='$NQ', allow_patterns=['pair/*','README.md'])"
[ -s jobs/rerank-v2/data/lists.jsonl ] || $PY src/rerank/build_lists_en.py $NQ/pair/train-00000-of-00001.parquet jobs/rerank-v2/data jobs/embed-v3/out $M/bge-reranker-v2-m3 6000 30 2>&1 | grep --line-buffered -v -i "warn\|Loading"
[ -s $O/train_meta.json ] || $PY src/rerank/train_rerank.py --lists jobs/rerank-data/out/lists.jsonl,jobs/rerank-v2/data/lists.jsonl \
  --base $M/mmBERT-small --init jobs/rerank-v1/out --out $O --steps 3000 --q 16 --g 16 --lr 2e-5 2>&1 | grep --line-buffered -v -i "warn\|Loading"
echo "== MIRACLReranking dev nDCG@10 (v1: native .806 roman .756 en .645; teacher .857/.560/.746; mMiniLM .774/.264/.680)"
$PY src/rerank/eval_rerank.py rerank-v2=$O 2>&1 | grep native
echo "[rerank-v2] end $(date -Is)"
