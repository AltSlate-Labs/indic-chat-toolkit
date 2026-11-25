#!/usr/bin/env bash
# embed-v2: fine-tune embed-v1 on +<=150k native sentences/lang (no romanization filter) + 500k v1 replay rows, to fix the
# student's weak Indic-side STS (v1 dev STS 47.1 vs teacher 53.2). Steps: build data (CPU, few min), teacher bge-m3 on unique
# targets (~1.8M texts / 1700/s = ~18 min), train 1 epoch bs 128 lr 3e-5 (~15k steps x 0.5 s = ~2.1 h, ckpt every 2000; resume
# by rerunning), then full dev eval. Each stage skips if its output exists.
set -euo pipefail
cd ~/agent/rnd; O=jobs/embed-v2/out; DD=jobs/embed-data-v2/out; M=~/Downloads/model; PY=envs/pii/bin/python
echo "[embed-v2] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
[ -s $DD/train.jsonl ] || $PY src/embed/build_native_extra.py jobs/embed-data/out $DD 150000 500000
[ -s $DD/targets.json ] || $PY src/embed/teacher.py $DD $M/bge-m3 2>&1 | grep --line-buffered -v -i warn
[ -s $O/train_meta.json ] || $PY src/embed/train_embed.py --data $DD --base jobs/embed-v1/out --init_head jobs/embed-v1/out/head.pt \
  --out $O --epochs 1 --bs 128 --lr 3e-5 --ckpt_every 2000 2>&1 | grep --line-buffered -v -i warn
echo "== full dev eval (v1: dakshina .996 flores .998 flores-ro .993 sts 47.1; teacher sts 53.2, l3cube 56.3)"
$PY src/embed/evaluate.py dev embed-v2=$O:mean_head 2>&1 | grep -v -i "warn\|Loading"
echo "[embed-v2] end $(date -Is)"
