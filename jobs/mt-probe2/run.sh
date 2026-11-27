#!/usr/bin/env bash
# Capacity probe: facebook/m2m100_418M (MIT, transformers.js-ready) 3k steps on data0, same recipe as the Marian probe
# (opus-mt-mul-en 77M: PHINC dev 51.4). Bigger student if it is clearly better.
cd ~/agent/rnd; M=~/Downloads/model; PY=envs/pii/bin/python; D=jobs/mt-data0/out; O=jobs/mt-probe2/out
echo "[mt-probe2] start $(date -Is)"
while tmux has -t job-mt-eval4 2>/dev/null; do sleep 30; done
docker ps --format '{{.Names}}' | grep -v '^rnd-' && { echo "other containers running; stop"; exit 1; } || true
[ -s $M/m2m100_418M/config.json ] || $PY -c "from huggingface_hub import snapshot_download as s; s('facebook/m2m100_418M', local_dir='$M/m2m100_418M', allow_patterns=['*.json','*.model','*.txt','pytorch_model.bin','*.safetensors'])" > /dev/null
ls $M/m2m100_418M
$PY src/mt/train_mt.py --base $M/m2m100_418M --src_lang hi --data $D/train.jsonl --dev jobs/mt-base/out/dev.jsonl --out $O/m2m --steps 3000 --lr 5e-5 --bs 64 2>&1 | grep --line-buffered -v -i warn | grep --line-buffered "params\|rows\|step [0-9]*000/\|DEV\|Error\|error"
echo "[mt-probe2] end $(date -Is)"
