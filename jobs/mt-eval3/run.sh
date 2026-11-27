#!/bin/bash
# Second dev set: COMI-LINGUA MT_train sample (500 sentences x roman/mixed), eval only. Test split untouched until release.
cd ~/agent/rnd; PY=envs/pii/bin/python; M=~/Downloads/model; F=jobs/mt-eval3/out/comi-dev.jsonl
echo "[mt-eval3] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-' && { echo "other containers running; stop"; exit 1; } || true
mkdir -p jobs/mt-eval3/out; $PY src/mt/make_comi_mt.py train $F 500
for spec in "copy copy x" "opus-mt-mul-en marian $M/opus-mt-mul-en" "nllb-600M nllb $M/nllb-200-distilled-600M" "mt-v1 marian jobs/mt-v1/out/model"; do
  set -- $spec; $PY src/mt/eval_mt.py $F $1-comi $2 $3 2>&1 | grep "chrF"
done
echo "[mt-eval3] end $(date -Is)"
