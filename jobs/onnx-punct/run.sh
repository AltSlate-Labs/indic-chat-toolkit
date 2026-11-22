#!/usr/bin/env bash
# ONNX export of the punctuation model: bash jobs/onnx-punct/run.sh v2  (fp32 + per-channel int8; scored through onnxruntime)
set -euo pipefail
cd ~/agent/rnd; V=$1; M=jobs/punct-$V/out/model; O=jobs/onnx-punct/out/$V; T=jobs/punct-v1/out; mkdir -p $O
. envs/pii/bin/activate
python3 -c "
import json
with open('$O/dev-text.jsonl','w') as f:
    for i,l in enumerate(open('jobs/punct-$V/out/dev.jsonl')):
        if i>=300: break
        f.write(json.dumps({'text':' '.join(json.loads(l)['words'])},ensure_ascii=False)+'\n')"
(cd src && python export_onnx.py ../$M ../$O ../$O/dev-text.jsonl)
python src/requant_onnx.py $O/model.onnx $O && mv $O/pc.onnx $O/model_quantized.onnx && rm -f $O/pc_head.onnx $O/head.onnx
ls -la $O
for T2 in wiki chat; do python src/wordlid/onnx_check.py $M $O $T/test-$T2.jsonl jobs/punct-$V/out/pred-$T2.jsonl 600; done
echo "[onnx-punct $V] end $(date -Is)"
