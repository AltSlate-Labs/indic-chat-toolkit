#!/usr/bin/env bash
# Release prep for a TN model: model-fix (tie=false + original byt5 tokenizer files), check fix == model, ONNX export
# (text2text-generation-with-past), per-channel int8 (onnx_check.py quantizes), direct-ORT greedy check on test-prog + test-multi.
# usage: bash jobs/onnx-tn/run.sh tn-v3
set -euo pipefail
V=$1; cd ~/agent/rnd
M=jobs/$V/out/model; F=jobs/$V/out/model-fix; O=jobs/onnx-tn/out/$V; P=jobs/$V/out
echo "[onnx-tn $V] start $(date -Is)"
mkdir -p $F $O
cp $M/model.safetensors $M/generation_config.json $F/
python3 - <<PY
import json; c=json.load(open("$M/config.json")); c["tie_word_embeddings"]=False; json.dump(c,open("$F/config.json","w"),indent=2)
PY
cp ~/Downloads/model/byt5-small/tokenizer_config.json ~/Downloads/model/byt5-small/special_tokens_map.json $F/
[ -f ~/Downloads/model/byt5-small/added_tokens.json ] && cp ~/Downloads/model/byt5-small/added_tokens.json $F/ || true
. envs/pii/bin/activate
python src/tn/predict_tn.py $F $P/test-multi.jsonl $P/pred-multi-fix.jsonl
cmp $P/pred-multi-fix.jsonl $P/pred-multi-$V.jsonl && echo "model-fix identical to model on test-multi"
deactivate; . envs/export/bin/activate
[ -s $O/onnx/encoder_model.onnx ] || optimum-cli export onnx --model $F --task text2text-generation-with-past $O/onnx
[ -s $O/onnx/encoder_model_quantized.onnx ] || (cd src/xlit && python onnx_check.py ../../$O/onnx ../../jobs/xlit-gap/out/testB.jsonl ../../jobs/xlit-v2/out/pred-B.jsonl 1 >/dev/null)
ls -la $O/onnx
for S in "" _quantized; do
  python src/tn/onnx_greedy_tn.py $O/onnx $P/test-prog.jsonl $P/pred-prog.jsonl 150 "$S"
  python src/tn/onnx_greedy_tn.py $O/onnx $P/test-multi.jsonl $P/pred-multi-$V.jsonl 100 "$S"
done
echo "[onnx-tn $V] end $(date -Is)"
