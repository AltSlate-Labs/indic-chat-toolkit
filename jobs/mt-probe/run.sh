#!/usr/bin/env bash
# Round-0 probe: english-to-hinglish (PHINC dev/test removed) + PHINC train -> fine-tune opus-mt-mul-en (77M) and mt5-small (300M)
# for 3000 steps each; PHINC dev chrF++. Baselines (mt-base): copy 42.2, nllb-600M 41.4, opus-mt-mul-en 35.6.
set -euo pipefail
cd ~/agent/rnd; M=~/Downloads/model; PY=envs/pii/bin/python; D=jobs/mt-data0/out; O=jobs/mt-probe/out
echo "[mt-probe] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
mkdir -p $D; [ -s $D/train.jsonl ] || $PY src/mt/build_data.py jobs/mt-base/out $D
[ -s $M/mt5-small/config.json ] || $PY -c "from huggingface_hub import snapshot_download as s; s('google/mt5-small', local_dir='$M/mt5-small', allow_patterns=['*.json','*.model','*.bin','*.safetensors'], ignore_patterns=['*tf_*','*flax*'])" > /dev/null
$PY src/mt/train_mt.py --base $M/opus-mt-mul-en --data $D/train.jsonl --dev jobs/mt-base/out/dev.jsonl --out $O/marian --steps 3000 --lr 1e-4 2>&1 | grep --line-buffered -v -i warn | grep --line-buffered "params\|rows\|step [0-9]*000/\|DEV\|Error\|error"
$PY src/mt/train_mt.py --base $M/mt5-small --data $D/train.jsonl --dev jobs/mt-base/out/dev.jsonl --out $O/mt5 --steps 3000 --lr 5e-4 2>&1 | grep --line-buffered -v -i warn | grep --line-buffered "params\|rows\|step [0-9]*000/\|DEV\|Error\|error"
echo "[mt-probe] end $(date -Is)"
