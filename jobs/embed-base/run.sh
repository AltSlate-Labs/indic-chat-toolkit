#!/usr/bin/env bash
# embed-base: distil bge-m3 into mmBERT-base (307M) in ONE mixed epoch drawn from the three small-model rounds (the small model's
# graded-similarity gap was not the loss: embed-v4probe nce_w 0.1 vs 1.0 equal). 4.0M rows -> 31.25k steps x 0.87 s = ~7.6 h.
# Each invocation trains at most 14000 steps (~3.4 h; chunk 1 used a 4 h timeout and lost 1.9k steps), checkpoint every 2000;
# rerun this script to resume. Eval when finished.
set -euo pipefail
cd ~/agent/rnd; O=jobs/embed-base/out; DD=jobs/embed-data-base/out; M=~/Downloads/model; PY=envs/pii/bin/python
echo "[embed-base] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
[ -s $DD/train.jsonl ] || $PY src/embed/build_mix.py $DD jobs/embed-data/out:*:1300000 jobs/embed-data-v2/out:native2:600000 \
  jobs/embed-data-v3/out:xl:1200000 jobs/embed-data-v3/out:xl-roman:600000 jobs/embed-data-v3/out:en:300000
if [ ! -s $O/train_meta.json ]; then
  $PY src/embed/train_embed.py --data $DD --base $M/mmBERT-base --out $O --epochs 1 --bs 128 --lr 5e-5 \
    --ckpt_every 2000 --chunk_steps 14000 2>&1 | grep --line-buffered -v -i warn || true
fi
if [ -s $O/train_meta.json ]; then
  echo "== full dev eval (small v3: dakshina .993 flores .998 flores-ro .995 sts 53.0 sts-en 81.4; teacher sts 53.2 sts-en 86.7)"
  $PY src/embed/evaluate.py dev embed-base=$O:mean_head 2>&1 | grep "mean"
else echo "[embed-base] chunk done, not finished: rerun to resume"; fi
echo "[embed-base] end $(date -Is)"
