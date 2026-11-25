#!/usr/bin/env bash
# embed-v1: distil bge-m3 into mmBERT-small (notes/embed-plan.md). 1 epoch of jobs/embed-data (2.27M rows), bs 128, lr 5e-5
# (head 1e-3), MSE + InfoNCE. 17.7k steps x 0.47 s = ~2.3 h; checkpoints every 2000 steps (resume by rerunning).
# Then: Dakshina DEV roman<->native recall@1 (same probe as the baselines).
set -euo pipefail
cd ~/agent/rnd; O=jobs/embed-v1/out; M=~/Downloads/model
echo "[embed-v1] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
[ -s $O/train_meta.json ] || envs/pii/bin/python src/embed/train_embed.py --data jobs/embed-data/out --base $M/mmBERT-small --out $O \
  --epochs 1 --bs 128 --lr 5e-5 --ckpt_every 2000 2>&1 | grep --line-buffered -v -i warn
echo "== Dakshina DEV roman<->native recall@1 (l3cube .828, bge-m3 .604, e5-small .478)"
envs/pii/bin/python src/embed/probe.py dev 1000 embed-v1=$O:mean_head 2>&1 | grep -v -i warn
echo "[embed-v1] end $(date -Is)"
