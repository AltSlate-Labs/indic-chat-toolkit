#!/usr/bin/env bash
# xlit-v0: ByT5-small (Apache-2.0) romanized -> native for 12 langs. Dakshina dev + lexicon words + Ornith chat. Eval on test A and B.
set -euo pipefail
cd ~/agent/rnd/src/xlit && . ../../envs/pii/bin/activate
export HF_HUB_DISABLE_PROGRESS_BARS=1
O=../../jobs/xlit-v0/out; T=../../jobs/xlit-gap/out; mkdir -p $O
echo "[xlit-v0] start $(date -Is)"
[ -d ~/Downloads/model/byt5-small ] || python -c "from huggingface_hub import snapshot_download as s; import os; s('google/byt5-small', local_dir=os.path.expanduser('~/Downloads/model/byt5-small'), allow_patterns=['*.json','*.safetensors','*.model','*.txt'])"
[ -s $O/train.jsonl ] || python build_data.py $O
head -3 $O/train.jsonl
[ -s $O/model/train_meta.json ] || python train_xlit.py --model ~/Downloads/model/byt5-small --train $O/train.jsonl --dev $O/dev.jsonl --out $O/model --epochs ${EPOCHS:-2} --bs 32 --lr 5e-4
for X in A B; do python predict.py $O/model $T/test$X.jsonl $O/pred-$X.jsonl; echo "== xlit-v0 test $X"; python metrics.py $T/test$X.jsonl $O/pred-$X.jsonl; done
echo "[xlit-v0] end $(date -Is)"
