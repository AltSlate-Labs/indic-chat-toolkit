#!/usr/bin/env bash
# punct-v1: punctuation restoration (O/COMMA/PERIOD/QUESTION after each word), mmBERT-small token classification; plan notes/punct-plan.md.
# Data src/punct/synth_punct.py (built pass 124). Tests: test-wiki (Dakshina TEST), test-chat (tn-gap test B, Qwen-written),
# test-ochat (held-out 10% of Ornith chat; same generator as training chat). Baseline: 1-800-BAD-CODE xlm-roberta (punctuators, CPU ONNX).
set -euo pipefail
cd ~/agent/rnd; O=jobs/punct-v1/out
echo "[punct-v1] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
. envs/pii/bin/activate
[ -s $O/model/config.json ] || (cd src/punct && python train_punct.py --model ~/Downloads/model/mmBERT-small --train ../../$O/train.jsonl \
   --dev ../../$O/dev.jsonl --out ../../$O/model --epochs 2 --bs 32 --lr 5e-5)
for T in wiki chat ochat; do python src/wordlid/predict_wordlid.py $O/model $O/test-$T.jsonl $O/pred-$T.jsonl; done
for T in wiki chat ochat; do echo "== test-$T"; A=""; [ -s $O/base-$T.jsonl ] && A="1800-xlmr=$O/base-$T.jsonl"
  python src/punct/score_punct.py $O/test-$T.jsonl $A punct-v1=$O/pred-$T.jsonl; done
echo "[punct-v1] end $(date -Is)"
