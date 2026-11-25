#!/usr/bin/env bash
# embed-eval: baselines on the held-out suite (src/embed/evaluate.py), SPLIT given as $1 (dev while developing; test once, at the end).
# Runs on the GPU alongside training (small: 6 encoders x ~25k sentences).
set -euo pipefail
cd ~/agent/rnd; M=~/Downloads/model; S=${1:-dev}
echo "[embed-eval $S] start $(date -Is)"
envs/pii/bin/python src/embed/evaluate.py $S \
  e5-small-118M=$M/multilingual-e5-small:mean:"query: " \
  MiniLM-L12-118M=$M/paraphrase-multilingual-MiniLM-L12-v2:mean \
  l3cube-indic-sbert-237M=$M/indic-sentence-similarity-sbert:mean \
  LaBSE-471M=$M/LaBSE:cls_dense \
  bge-m3-568M=$M/bge-m3:cls 2>&1 | grep -v -i warn
echo "[embed-eval $S] end $(date -Is)"
