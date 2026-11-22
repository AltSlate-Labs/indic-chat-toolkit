#!/usr/bin/env bash
# wordlid-v1: word-level LID (12 Indic languages + en + other), mmBERT-small token classification; plan notes/wordlid-plan.md.
# Data: src/wordlid/synth_wordlid.py (built in pass 116; 198k rows, a quarter Indic+Indic mixes). 2 epochs.
# Tests: test-mix (Dakshina TEST split only, exact labels) and COMI-LINGUA LID test (human, hi/en; metric fixed before results:
# Latin-script hi/en word accuracy) vs PhysicsWallahAI/muril-hinglish-lid (0.875 / all hi/en 0.888, jobs/wordlid-gap).
set -euo pipefail
cd ~/agent/rnd; O=jobs/wordlid-v1/out; G=jobs/wordlid-gap/out
echo "[wordlid-v1] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
. envs/pii/bin/activate
[ -s $O/test-mix.jsonl ] || python src/wordlid/synth_wordlid.py $O 200000
[ -s $O/model/config.json ] || (cd src/wordlid && python train_wordlid.py --model ~/Downloads/model/mmBERT-small --train ../../$O/train.jsonl \
   --dev ../../$O/dev.jsonl --out ../../$O/model --epochs 2 --bs 32 --lr 5e-5)
python src/wordlid/predict_wordlid.py $O/model $O/test-mix.jsonl $O/pred-mix.jsonl
python src/wordlid/predict_wordlid.py $O/model $G/comi-test.jsonl $O/pred-comi.jsonl comi
echo "== test-mix"; python src/wordlid/score_mix.py $O/test-mix.jsonl wordlid-v1=$O/pred-mix.jsonl
echo "== COMI-LINGUA"; python src/wordlid/score_comi.py $G/comi-test.jsonl muril-hinglish=$G/pred-muril.jsonl wordlid-v1=$O/pred-comi.jsonl
echo "[wordlid-v1] end $(date -Is)"
