#!/usr/bin/env bash
# Feasibility probe for a small Hinglish/romanized -> English translator: split PHINC, score ungated baselines on PHINC DEV.
set -euo pipefail
cd ~/agent/rnd; M=~/Downloads/model; PY=envs/pii/bin/python; D=jobs/mt-base/out
echo "[mt-base] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
mkdir -p $D; [ -s $D/test.jsonl ] || $PY src/mt/split_phinc.py $D
for r in Helsinki-NLP/opus-mt-hi-en Helsinki-NLP/opus-mt-mul-en facebook/nllb-200-distilled-600M; do
  n=${r#*/}; [ -s $M/$n/config.json ] || $PY -c "from huggingface_hub import snapshot_download as s; s('$r', local_dir='$M/$n', allow_patterns=['*.json','*.spm','*.model','*.txt','*.safetensors','*.bin'], ignore_patterns=['*tf_model*','*flax*','*rust*'])" > /dev/null
done
$PY src/mt/eval_mt.py $D/dev.jsonl copy copy 2>&1 | grep chrF
$PY src/mt/eval_mt.py $D/dev.jsonl opus-mt-hi-en marian $M/opus-mt-hi-en 2>&1 | grep chrF
$PY src/mt/eval_mt.py $D/dev.jsonl opus-mt-mul-en marian $M/opus-mt-mul-en 2>&1 | grep chrF
$PY src/mt/eval_mt.py $D/dev.jsonl nllb-600M nllb $M/nllb-200-distilled-600M 2>&1 | grep chrF
echo "[mt-base] end $(date -Is)"
