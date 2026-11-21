#!/usr/bin/env bash
# Release prep for xlit-v1: fixed config (tie=false) + original byt5 tokenizer files, ONNX export, per-channel int8, direct-ORT check.
set -euo pipefail
cd ~/agent/rnd/src/xlit
M=../../jobs/xlit-v1/out/model; F=../../jobs/xlit-v1/out/model-fix; O=../../jobs/onnx-xlit/out/v1; T=../../jobs/xlit-gap/out
echo "[onnx-xlit-v1] start $(date -Is)"
mkdir -p $F $O
cp $M/model.safetensors $M/generation_config.json $F/
python3 - <<PY
import json; c=json.load(open("$M/config.json")); c["tie_word_embeddings"]=False; json.dump(c,open("$F/config.json","w"),indent=2)
PY
cp ~/Downloads/model/byt5-small/tokenizer_config.json ~/Downloads/model/byt5-small/special_tokens_map.json $F/
[ -f ~/Downloads/model/byt5-small/added_tokens.json ] && cp ~/Downloads/model/byt5-small/added_tokens.json $F/ || true
. ../../envs/pii/bin/activate
python predict.py $F $T/testB.jsonl ../../jobs/xlit-v1/out/pred-B-fix.jsonl
cmp ../../jobs/xlit-v1/out/pred-B-fix.jsonl ../../jobs/xlit-v1/out/pred-B.jsonl && echo "model-fix identical to model on B"
deactivate; . ../../envs/export/bin/activate
[ -s $O/onnx/encoder_model.onnx ] || optimum-cli export onnx --model $F --task text2text-generation-with-past $O/onnx
python onnx_check.py $O/onnx $T/testB.jsonl ../../jobs/xlit-v1/out/pred-B.jsonl 5
python onnx_greedy.py $O/onnx $T/testB.jsonl ../../jobs/xlit-v1/out/pred-B.jsonl 60 ""
python onnx_greedy.py $O/onnx $T/testB.jsonl ../../jobs/xlit-v1/out/pred-B.jsonl 60 _quantized
echo "[onnx-xlit-v1] end $(date -Is)"
