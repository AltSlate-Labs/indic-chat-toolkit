#!/usr/bin/env bash
# mt-v1: opus-mt-mul-en (77M) on Hinglish data0 x2 + NLLB native/romanized Indic->English (40k+40k per language), 30k steps
# (~65 min at 0.13 s/step, ckpt every 5k). PHINC dev (mt-probe marian 3k steps: 51.4) + multi-language romanized dev.
set -euo pipefail
cd ~/agent/rnd; PY=envs/pii/bin/python; M=~/Downloads/model; D=jobs/mt-data1/out; O=jobs/mt-v1/out
echo "[mt-v1] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
[ -s $D/train.jsonl ] || $PY src/mt/build_mix_mt.py jobs/mt-data0/out jobs/embed-data-v3/out/train.jsonl $D 40000 2
[ -s $O/train_meta.json ] || $PY src/mt/train_mt.py --base $M/opus-mt-mul-en --data $D/train.jsonl --dev jobs/mt-base/out/dev.jsonl \
  --out $O --steps 30000 --lr 1e-4 --ckpt_every 5000 2>&1 | grep --line-buffered -v -i warn | grep --line-buffered "params\|rows\|resumed\|step [0-9]*000/\|DEV\|Error\|error"
$PY src/mt/eval_mt.py $D/dev-xl.jsonl mt-v1-xl marian $O/model 2>&1 | grep chrF
$PY src/mt/eval_mt.py $D/dev-xl.jsonl copy-xl copy 2>&1 | grep chrF
echo "[mt-v1] end $(date -Is)"
