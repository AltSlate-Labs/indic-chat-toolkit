#!/usr/bin/env bash
# Full IndicLID (FTN/FTR + BERT fallback, official thresholds) vs GlotLID vs lid-v3 on test A + B.
set -euo pipefail
cd ~/agent/rnd/src/lid
O=../../jobs/lid-indiclid/out
A=../../jobs/lid-gap/out/dakshina-test.jsonl; B=../../jobs/lid-testb/out/testb.jsonl
. ../../envs/lid/bin/activate
python indiclid_full.py ft $A $B $O/indiclid-ft.json
deactivate; . ../../envs/pii/bin/activate
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HOME=~/Downloads/model/hf-home-indiclid
python indiclid_full.py bert $O/indiclid-ft.json $O/indiclid-full-cache.json
python compare.py ../../jobs/lid-v3/out/model $A $B ../../jobs/lid-v3/out/glot-cache.json $O/indiclid-full-cache.json
echo "[lid-indiclid3] end $(date -Is)"
