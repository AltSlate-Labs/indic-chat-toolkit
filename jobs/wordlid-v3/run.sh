#!/usr/bin/env bash
# wordlid-v3: v2 passed COMI (.900 vs muril .875) but hand-written checks failed: English loanwords inside romanized Indic labelled
# Indic (phone, bike: Dakshina convention), kn chat labelled ml, romanized Hindi labelled ur. Changes: English-word convention for
# Latin words in romanized segments (src/wordlid/envocab.py: Wikipedia-vs-Indic frequency ratio + COMI loanwords, lowercase only),
# + real romanized chat segments (Ornith chat, MIT; Hinglish-Everyday-Conversations, MIT). test-mix gold relabelled with the same
# convention (test-mix-v3.jsonl); v1/v2 rescored on it.
set -euo pipefail
cd ~/agent/rnd; O=jobs/wordlid-v3/out; G=jobs/wordlid-gap/out; mkdir -p $O
echo "[wordlid-v3] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
. envs/pii/bin/activate
CH=jobs/lid-chat/out/chat-train.jsonl,jobs/lid-chat4/out/chat-weak.jsonl
[ -s $O/train.jsonl ] || ENVOCAB=$O/envocab.json CHAT=$CH COMI_TRAIN=$G/comi-train.jsonl COMI_TEST=$G/comi-test.jsonl NO_TEST=1 \
   python src/wordlid/synth_wordlid.py $O 200000
[ -s $O/model/config.json ] || (cd src/wordlid && python train_wordlid.py --model ~/Downloads/model/mmBERT-small --train ../../$O/train.jsonl \
   --dev ../../$O/dev.jsonl --out ../../$O/model --epochs 2 --bs 32 --lr 5e-5)
python src/wordlid/predict_wordlid.py $O/model $O/test-mix-v3.jsonl $O/pred-mix.jsonl
for V in v1 v2; do python src/wordlid/predict_wordlid.py jobs/wordlid-$V/out/model $O/test-mix-v3.jsonl $O/pred-mix-$V.jsonl; done
python src/wordlid/predict_wordlid.py $O/model $G/comi-test.jsonl $O/pred-comi.jsonl comi
echo "== test-mix-v3 (gold with the English-word convention)"
python src/wordlid/score_mix.py $O/test-mix-v3.jsonl wordlid-v1=$O/pred-mix-v1.jsonl wordlid-v2=$O/pred-mix-v2.jsonl wordlid-v3=$O/pred-mix.jsonl
echo "== COMI-LINGUA"; python src/wordlid/score_comi.py $G/comi-test.jsonl muril-hinglish=$G/pred-muril.jsonl wordlid-v2=jobs/wordlid-v2/out/pred-comi.jsonl wordlid-v3=$O/pred-comi.jsonl
echo "== sanity (hand-written, dev only)"; python src/wordlid/show.py $O/model src/wordlid/sanity.txt 2>/dev/null | grep -v "Loading"
echo "[wordlid-v3] end $(date -Is)"
