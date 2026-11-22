#!/usr/bin/env bash
# wordlid-v2: v1 got test-mix .960 but COMI-LINGUA Latin hi/en only .395 (muril-hinglish .875): Latin English words next to
# Devanagari were labelled hi (14.8k), romanized Hindi labelled ur (3.5k). Changes: + COMI-LINGUA LID TRAIN split (human labels,
# CC-BY-4.0; test split held out, exact-sentence overlap removed), + English Wikipedia sentences, half of the English-insert rows on
# native-script bases. Same test-mix file as v1. NOTE for the card: unlike muril-hinglish-lid, v2 trains on COMI's train split.
set -euo pipefail
cd ~/agent/rnd; O=jobs/wordlid-v2/out; G=jobs/wordlid-gap/out; V1=jobs/wordlid-v1/out; mkdir -p $O
echo "[wordlid-v2] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
. envs/pii/bin/activate
[ -s $G/comi-train.jsonl ] || python3 src/wordlid/comi_test.py ~/Downloads/data/comi-lingua/LID_train.csv $G/comi-train.jsonl
[ -s $O/train.jsonl ] || COMI_TRAIN=$G/comi-train.jsonl COMI_TEST=$G/comi-test.jsonl NO_TEST=1 python src/wordlid/synth_wordlid.py $O 200000
[ -s $O/model/config.json ] || (cd src/wordlid && python train_wordlid.py --model ~/Downloads/model/mmBERT-small --train ../../$O/train.jsonl \
   --dev ../../$O/dev.jsonl --out ../../$O/model --epochs 2 --bs 32 --lr 5e-5)
python src/wordlid/predict_wordlid.py $O/model $V1/test-mix.jsonl $O/pred-mix.jsonl
python src/wordlid/predict_wordlid.py $O/model $G/comi-test.jsonl $O/pred-comi.jsonl comi
echo "== test-mix"; python src/wordlid/score_mix.py $V1/test-mix.jsonl wordlid-v1=$V1/pred-mix.jsonl wordlid-v2=$O/pred-mix.jsonl
echo "== COMI-LINGUA"; python src/wordlid/score_comi.py $G/comi-test.jsonl muril-hinglish=$G/pred-muril.jsonl wordlid-v1=$V1/pred-comi.jsonl wordlid-v2=$O/pred-comi.jsonl
echo "[wordlid-v2] end $(date -Is)"
