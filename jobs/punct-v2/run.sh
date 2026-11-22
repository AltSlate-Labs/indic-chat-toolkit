#!/usr/bin/env bash
# punct-v2: v1 (mmBERT-small) ~ baseline on test-wiki (.625 vs .610 on rows both align) but worse on test-chat (.693 vs .809, commas)
# and on ta/te/hi/kn wiki. Baseline is XLM-R base (~270M) -> mmBERT-base (307M), more wiki rows (25k/lang), same tests.
set -euo pipefail
cd ~/agent/rnd; O=jobs/punct-v2/out; V1=jobs/punct-v1/out; mkdir -p $O
echo "[punct-v2] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
. envs/pii/bin/activate
[ -s $O/train.jsonl ] || WIKI_ROWS=25000 python src/punct/synth_punct.py $O
for T in wiki chat ochat; do cmp -s $O/test-$T.jsonl $V1/test-$T.jsonl && echo "test-$T identical to v1" || echo "WARNING test-$T differs from v1"; done
[ -s $O/model/config.json ] || (cd src/punct && GPU_MEM_FRAC=0.6 python train_punct.py --model ~/Downloads/model/mmBERT-base --train ../../$O/train.jsonl \
   --dev ../../$O/dev.jsonl --out ../../$O/model --epochs 1.5 --bs 32 --lr 4e-5)
for T in wiki chat ochat; do python src/wordlid/predict_wordlid.py $O/model $V1/test-$T.jsonl $O/pred-$T.jsonl; done
for T in wiki chat ochat; do echo "== test-$T"; python src/punct/score_punct.py $V1/test-$T.jsonl 1800-xlmr=$V1/base-$T.jsonl punct-v1=$V1/pred-$T.jsonl punct-v2=$O/pred-$T.jsonl; done
echo "[punct-v2] end $(date -Is)"
