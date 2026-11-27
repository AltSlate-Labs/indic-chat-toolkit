#!/usr/bin/env bash
# mt-v3b: continue mt-v3 with native Telugu mapped to Kannada script (M2M100 vocab lacks Telugu: 41% <unk>; zero-shot map lifts FLORES
# te native 17.7 -> 29.2). Data = data2 + its 40k native-Telugu rows x4 extra; 4k steps bs 64 lr 3e-5 (~42 min at 0.62 s/step).
set -euo pipefail
cd ~/agent/rnd; PY=envs/pii/bin/python; D=jobs/mt-data2/out; O=jobs/mt-v3b/out; B=jobs/mt-base/out
echo "[mt-v3b] start $(date -Is)"
while tmux has -t job-mt-eval5 2>/dev/null; do sleep 30; done
docker ps --format '{{.Names}}' | grep -v '^rnd-' && { echo "other containers running; stop"; exit 1; } || true
mkdir -p $O
[ -s $O/train.jsonl ] || python3 -c "
import json, random
rows = open('$D/train.jsonl', encoding='utf-8').readlines()
te = [l for l in rows if any(0xC00 <= ord(c) <= 0xC7F for c in json.loads(l)['src'])]
out = rows + te * 4; random.Random(5).shuffle(out)
open('$O/train.jsonl', 'w', encoding='utf-8').writelines(out); print('te rows', len(te), 'total', len(out))"
[ -s $O/train_meta.json ] || $PY src/mt/train_mt.py --base jobs/mt-v3/out/model --src_lang auto --data $O/train.jsonl --dev $B/dev.jsonl \
  --out $O --steps 4000 --lr 3e-5 --ckpt_every 2000 2>&1 | grep --line-buffered -v -i warn | grep --line-buffered "params\|rows\|resumed\|step [0-9]*000/\|DEV\|Error\|error"
$PY src/mt/eval_mt.py jobs/mt-data1/out/dev-xl.jsonl mt-v3b-xl m2m $O/model 2>&1 | grep chrF
$PY src/mt/eval_mt.py jobs/mt-eval1/out/flores-dev.jsonl mt-v3b-flores m2m $O/model 2>&1 | grep chrF
$PY src/mt/eval_mt.py jobs/mt-eval3/out/comi-dev.jsonl mt-v3b-comi m2m $O/model 2>&1 | grep chrF
cp $O/pred-dev.jsonl $B/pred-mt-v3b-phinc.jsonl
rm -f $O/train.jsonl $O/ckpt.pt
echo "[mt-v3b] end $(date -Is)"
