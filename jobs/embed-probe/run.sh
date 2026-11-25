#!/usr/bin/env bash
# embed-probe: how well do existing small embedders match human-romanized Indic sentences to their native-script originals?
# Dakshina rejoined DEV split (test split kept untouched for a future model), 1000 pairs per language, recall@1 both directions.
set -euo pipefail
cd ~/agent/rnd; M=~/Downloads/model
echo "[embed-probe] start $(date -Is)"
envs/pii/bin/python - <<PY
from huggingface_hub import snapshot_download
import os
for r in ["intfloat/multilingual-e5-small", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
          "l3cube-pune/indic-sentence-similarity-sbert", "BAAI/bge-m3", "sentence-transformers/LaBSE"]:
    d = os.path.expanduser("~/Downloads/model/" + r.split("/")[1])
    snapshot_download(r, local_dir=d, allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "pytorch_model.bin", "sentencepiece*", "*.py"]); print("ok", r)
PY
envs/pii/bin/python src/embed/probe.py dev 1000 \
  l3cube-indic-sbert-237M=$M/indic-sentence-similarity-sbert:mean \
  LaBSE-471M=$M/LaBSE:cls_dense \
  bge-m3-568M=$M/bge-m3:cls \
  mmBERT-small-raw-140M=$M/mmBERT-small:mean 2>&1 | grep -v -i "warn"
echo "[embed-probe] end $(date -Is)"
