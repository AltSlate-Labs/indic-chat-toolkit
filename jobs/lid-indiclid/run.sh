#!/usr/bin/env bash
# Missed baseline: AI4Bharat IndicLID (MIT; HF ai4bharat/IndicLID-FTR, -FTN fastText; BERT stage not run).
# Same test A + B as lid-v3; prints GlotLID, IndicLID-FT and lid-v3 side by side.
set -euo pipefail
cd ~/agent/rnd/src/lid
O=../../jobs/lid-indiclid/out; mkdir -p $O
echo "[lid-indiclid] start $(date -Is)"
A=../../jobs/lid-gap/out/dakshina-test.jsonl; B=../../jobs/lid-testb/out/testb.jsonl
. ../../envs/lid/bin/activate
python compare.py indiclid $A $B $O/indiclid-cache.json > /dev/null
deactivate; . ../../envs/pii/bin/activate
python compare.py ../../jobs/lid-v3/out/model $A $B ../../jobs/lid-v3/out/glot-cache.json $O/indiclid-cache.json
echo "[lid-indiclid] end $(date -Is)"
