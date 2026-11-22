#!/usr/bin/env bash
# ONNX export of a word-level LID model: bash jobs/onnx-wordlid/run.sh v3
# fp32 + per-channel dynamic int8 (as for the PII and LID releases), then word accuracy through onnxruntime vs PyTorch.
set -euo pipefail
cd ~/agent/rnd; V=$1; M=jobs/wordlid-$V/out/model; O=jobs/onnx-wordlid/out/$V; T=jobs/wordlid-v3/out; G=jobs/wordlid-gap/out; mkdir -p $O
. envs/pii/bin/activate
python3 -c "
import json
with open('$O/dev-text.jsonl','w') as f:
    for i,l in enumerate(open('jobs/wordlid-$V/out/dev.jsonl')):
        if i>=500: break
        f.write(json.dumps({'text':' '.join(json.loads(l)['words'])},ensure_ascii=False)+'\n')"
(cd src && python export_onnx.py ../$M ../$O ../$O/dev-text.jsonl)
python src/requant_onnx.py $O/model.onnx $O && mv $O/pc.onnx $O/model_quantized.onnx && rm -f $O/pc_head.onnx $O/head.onnx
ls -la $O
python src/wordlid/onnx_check.py $M $O $T/test-mix-v3.jsonl $T/pred-mix.jsonl 1000
python src/wordlid/onnx_check.py $M $O $G/comi-test.jsonl $T/pred-comi.jsonl 1000 comi
echo "[onnx-wordlid $V] end $(date -Is)"
