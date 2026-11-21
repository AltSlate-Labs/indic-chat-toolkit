#!/usr/bin/env bash
# IndicLID-FT restricted to our label set (generous baseline) + out-of-scope probe on Bhasha-Abhijnaanam manual roman.
set -euo pipefail
cd ~/agent/rnd/src/lid
O=../../jobs/lid-indiclid/out
A=../../jobs/lid-gap/out/dakshina-test.jsonl; B=../../jobs/lid-testb/out/testb.jsonl
. ../../envs/lid/bin/activate
IND_RESTRICT=1 python compare.py indiclid $A $B $O/indiclid-r-cache.json > /dev/null
deactivate; . ../../envs/pii/bin/activate
python compare.py ../../jobs/lid-v3/out/model $A $B ../../jobs/lid-v3/out/glot-cache.json $O/indiclid-r-cache.json | grep indiclid
python out_of_scope.py ../../jobs/lid-v3/out/model ~/Downloads/data/bhasha-abhijnaanam/bhasha-abhijnaanam.json
echo "[lid-indiclid2] end $(date -Is)"
