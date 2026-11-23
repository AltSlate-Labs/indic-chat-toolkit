#!/usr/bin/env bash
# itn-v3 on test-B-clean (already NFC) and on the card's example inputs.
set -euo pipefail
cd ~/agent/rnd; O=jobs/nfc-v5/out; E=jobs/itn-eval/out
. envs/pii/bin/activate
python src/tn/predict_tn.py $O/itn-v3 $E/test-B-clean.jsonl $O/itn-pred-Bclean.jsonl
python src/tn/predict_tn.py $O/itn-v3 $O/itn-examples.jsonl $O/itn-pred-examples.jsonl
deactivate
echo "== test-B-clean"; python3 src/itn/score_itn.py $E/test-B-clean.jsonl itn-v2=$E/itn-v2/clean-itn-v2.jsonl itn-v3=$O/itn-pred-Bclean.jsonl
paste -d'\t' $O/itn-examples.jsonl $O/itn-pred-examples.jsonl | python3 -c "
import sys,json
for l in sys.stdin:
    a,b=l.rstrip('\n').split('\t'); a,b=json.loads(a),json.loads(b); print('SAME' if a['tgt']==b['pred'] else 'DIFF', a['lang'], b['pred'][:120])"
echo "[itn-b] end"
