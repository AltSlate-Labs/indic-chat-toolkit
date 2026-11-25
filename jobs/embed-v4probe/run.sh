#!/usr/bin/env bash
# embed-v4probe: does the in-batch contrastive term hurt graded similarity? Continue embed-v3 for 4000 steps on the v3 data
# (lr 2e-5, bs 128) twice: (a) nce_w 0.1 (MSE-dominant), (b) nce_w 1.0 (control, same as v3). 2 x 4000 x 0.47 s = ~63 min,
# then dev eval of both (dev splits only; sts-en = STSb validation; en-xx STS is test-only -> reported, not used alone).
set -euo pipefail
cd ~/agent/rnd; PY=envs/pii/bin/python; DD=jobs/embed-data-v3/out; B=jobs/embed-v3/out
echo "[embed-v4probe] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
for v in nce01:0.1 nce10:1.0; do
  n=${v%%:*}; w=${v##*:}; O=jobs/embed-v4probe/$n
  [ -s $O/train_meta.json ] || $PY src/embed/train_embed.py --data $DD --base $B --init_head $B/head.pt --out $O \
    --max_steps 4000 --bs 128 --lr 2e-5 --nce_w $w --ckpt_every 2000 2>&1 | grep --line-buffered -v -i warn | grep --line-buffered "dev cos\|saved\|train "
done
echo "== dev eval (v3: dakshina .993 flores .998 flores-ro .995 sts 53.0 sts-en 81.4)"
$PY src/embed/evaluate.py dev nce01=jobs/embed-v4probe/nce01:mean_head nce10=jobs/embed-v4probe/nce10:mean_head 2>&1 | grep "mean"
echo "[embed-v4probe] end $(date -Is)"
