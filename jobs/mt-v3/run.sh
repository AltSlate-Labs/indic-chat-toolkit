#!/usr/bin/env bash
# mt-v3: facebook/m2m100_418M (MIT) on data2 (data1 + filtered Ornith Hinglish targets x2; 1.43M rows), per-row source-language
# token (kind suffix, Hinglish = hi), 20k steps bs 64 lr 5e-5 (~4.2 h at 0.76 s/step), checkpoint every 2.5k (resumable).
# Probe (3k steps, data0 only): PHINC dev 55.2 vs Marian probe 51.4 and mt-v2 55.1.
set -euo pipefail
cd ~/agent/rnd; PY=envs/pii/bin/python; M=~/Downloads/model; D=jobs/mt-data2/out; O=jobs/mt-v3/out; B=jobs/mt-base/out
echo "[mt-v3] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-' && { echo "other containers running; stop"; exit 1; } || true
[ -s $O/train_meta.json ] || $PY src/mt/train_mt.py --base $M/m2m100_418M --src_lang auto --data $D/train.jsonl --dev $B/dev.jsonl \
  --out $O --steps 20000 --lr 5e-5 --ckpt_every 2500 2>&1 | grep --line-buffered -v -i warn | grep --line-buffered "params\|rows\|resumed\|step [0-9]*000/\|DEV\|Error\|error"
$PY src/mt/eval_mt.py jobs/mt-data1/out/dev-xl.jsonl mt-v3-xl m2m $O/model 2>&1 | grep chrF
$PY src/mt/eval_mt.py jobs/mt-eval1/out/flores-dev.jsonl mt-v3-flores m2m $O/model 2>&1 | grep chrF
$PY src/mt/eval_mt.py jobs/mt-eval3/out/comi-dev.jsonl mt-v3-comi m2m $O/model 2>&1 | grep chrF
cp $O/pred-dev.jsonl $B/pred-mt-v3-phinc.jsonl
echo "[mt-v3] end $(date -Is)"
