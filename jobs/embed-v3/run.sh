#!/usr/bin/env bash
# embed-v3: fine-tune embed-v2 with cross-lingual distillation on NLLB en-xx (120k pairs/lang): xx -> teacher(en), 50% romanized
# xx -> teacher(en), en -> teacher(en), + 600k replay rows of v2 data. ~3.0M rows -> ~23k steps x 0.48 s = ~3.1 h (ckpt 2000,
# resume by rerunning). Data build is CPU and runs first; the GPU stages wait until job-embed-v2 has ended.
set -euo pipefail
cd ~/agent/rnd; O=jobs/embed-v3/out; DD=jobs/embed-data-v3/out; M=~/Downloads/model; PY=envs/pii/bin/python
echo "[embed-v3] start $(date -Is)"
while tmux has-session -t job-nllb-fetch 2>/dev/null; do sleep 60; done
[ "$(ls ~/Downloads/data/nllb-enxx/*.tsv | wc -l)" = 10 ] || { echo "nllb fetch incomplete; stop"; exit 1; }
[ -s $DD/train.jsonl ] || $PY src/embed/build_xling.py jobs/embed-data-v2/out ~/Downloads/data/nllb-enxx $DD 600000 0.5
while tmux has-session -t job-embed-v2 2>/dev/null; do sleep 60; done
[ -s jobs/embed-v2/out/train_meta.json ] || { echo "v2 did not finish; stop"; exit 1; }
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
[ -s $DD/targets.json ] || $PY src/embed/teacher.py $DD $M/bge-m3 2>&1 | grep --line-buffered -v -i warn
[ -s $O/train_meta.json ] || $PY src/embed/train_embed.py --data $DD --base jobs/embed-v2/out --init_head jobs/embed-v2/out/head.pt \
  --out $O --epochs 1 --bs 128 --lr 3e-5 --ckpt_every 2000 2>&1 | grep --line-buffered -v -i warn
echo "== full dev eval (v1 sts 47.1, v2-s4000 49.1; teacher 53.2, l3cube 56.3)"
$PY src/embed/evaluate.py dev embed-v3=$O:mean_head 2>&1 | grep -v -i "warn\|Loading"
echo "[embed-v3] end $(date -Is)"
