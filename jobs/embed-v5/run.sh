#!/usr/bin/env bash
# embed-v5: does caption/everyday English text fix the embedder's English graded similarity (STSb dev 81.4 vs 85-87)?
# SNLI sentences (<=500k, eval sentences removed) -> teacher vectors; mix with 1.0M replayed v3 rows; continue embed-v3 for one
# epoch (~11.7k steps x 0.47 s = ~1.5 h, lr 2e-5). Dev eval after (sts-en = STSb validation). Not on STSb test.
set -euo pipefail
cd ~/agent/rnd; M=~/Downloads/model; PY=envs/pii/bin/python; NL=jobs/embed-data-nli/out; DD=jobs/embed-data-v5/out; O=jobs/embed-v5/out
echo "[embed-v5] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
[ -s ~/Downloads/data/snli/plain_text/train-00000-of-00001.parquet ] || $PY -c "from huggingface_hub import snapshot_download as s; s('stanfordnlp/snli', repo_type='dataset', local_dir='$HOME/Downloads/data/snli', allow_patterns=['plain_text/*','README.md'])"
[ -s $NL/train.jsonl ] || $PY src/embed/build_nli_text.py $NL jobs/embed-data-v3/out 500000
[ -s $NL/targets.json ] || $PY src/embed/teacher.py $NL $M/bge-m3 2>&1 | grep --line-buffered -v -i warn | tail -2
[ -s $DD/train.jsonl ] || $PY src/embed/build_mix.py $DD jobs/embed-data-v3/out:*:1000000 $NL:*:500000
[ -s $O/train_meta.json ] || $PY src/embed/train_embed.py --data $DD --base jobs/embed-v3/out --init_head jobs/embed-v3/out/head.pt \
  --out $O --epochs 1 --bs 128 --lr 2e-5 --ckpt_every 2000 2>&1 | grep --line-buffered -v -i "warn\|Loading" | grep --line-buffered "dev cos\|train \|saved\|step [0-9]*000/"
echo "== dev eval (v3: dakshina .993 flores .998 flores-ro .995 sts 53.0 sts-en 81.4)"
$PY src/embed/evaluate.py dev embed-v5=$O:mean_head 2>&1 | grep "mean"
echo "[embed-v5] end $(date -Is)"
