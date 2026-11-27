#!/usr/bin/env bash
# mt-v2: continue mt-v1 on data1 + Ornith English targets for the Hinglish sources (x2), 20k steps lr 5e-5 (~55 min).
set -euo pipefail
cd ~/agent/rnd; PY=envs/pii/bin/python; D=jobs/mt-data2/out; O=jobs/mt-v2/out
echo "[mt-v2] start $(date -Is)"
while tmux has -t job-mt-orn1 2>/dev/null; do sleep 30; done
docker ps --format '{{.Names}}' | grep -v '^rnd-' && { echo "other containers running; stop"; exit 1; } || true
[ -s $D/train.jsonl ] || $PY src/mt/build_mix_mt2.py jobs/mt-data1/out/train.jsonl jobs/mt-orn1/out/train-ornith.jsonl $D 2
[ -s $O/train_meta.json ] || $PY src/mt/train_mt.py --base jobs/mt-v1/out/model --data $D/train.jsonl --dev jobs/mt-base/out/dev.jsonl \
  --out $O --steps 20000 --lr 5e-5 --ckpt_every 5000 2>&1 | grep --line-buffered -v -i warn | grep --line-buffered "params\|rows\|resumed\|step [0-9]*000/\|DEV\|Error\|error"
$PY src/mt/eval_mt.py jobs/mt-data1/out/dev-xl.jsonl mt-v2-xl marian $O/model 2>&1 | grep chrF
$PY src/mt/eval_mt.py jobs/mt-eval1/out/flores-dev.jsonl mt-v2-flores marian $O/model 2>&1 | grep chrF
$PY src/mt/eval_mt.py jobs/mt-eval3/out/comi-dev.jsonl mt-v2-comi marian $O/model 2>&1 | grep chrF
cp $O/pred-dev.jsonl jobs/mt-base/out/pred-mt-v2-phinc.jsonl
echo "[mt-v2] end $(date -Is)"
