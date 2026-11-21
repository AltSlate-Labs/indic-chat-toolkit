#!/usr/bin/env bash
# xlit-gap: is there a gap for small romanized->native transliteration of Indic chat? Tests from held-out LID data.
set -euo pipefail
cd ~/agent/rnd/src/xlit && . ../../envs/pii/bin/activate
O=../../jobs/xlit-gap/out; mkdir -p $O
python make_tests.py $O
for T in A B; do for MODE in hi_en hi; do
  python run_unimetype.py $O/test$T.jsonl $O/unime-$T-$MODE.jsonl $MODE 2>&1 | tail -1
  echo "== UnimeType-0.8B test $T (hi) mode $MODE"; python metrics.py $O/unime-$T-$MODE.jsonl $O/unime-$T-$MODE.jsonl
done; done
python - <<PY
import json
for T in "AB":
    xs=[json.loads(l) for l in open("$O/test%s.jsonl"%T)]
    with open("$O/identity-%s.jsonl"%T,"w") as f:
        for x in xs: f.write(json.dumps({"pred": x["roman"]})+"\n")
PY
for T in A B; do echo "== identity (copy roman) test $T"; python metrics.py $O/test$T.jsonl $O/identity-$T.jsonl; done
echo "[xlit-gap] end $(date -Is)"
