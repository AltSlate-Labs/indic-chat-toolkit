#!/usr/bin/env bash
# Release prep for tn-v5 / itn-v3 (NFC retrains): lay out jobs/<V>/out like earlier versions, then jobs/onnx-tn/run.sh <V>.
set -euo pipefail
cd ~/agent/rnd; N=jobs/nfc-v5/out
nfc() { python3 -c "
import json,sys,unicodedata as u
for l in open(sys.argv[1],encoding='utf-8'):
    x=json.loads(l); print(json.dumps({k:(u.normalize('NFC',v) if isinstance(v,str) else v) for k,v in x.items()},ensure_ascii=False))" $1 > $2; }
for pair in "tn-v5 tn jobs/tn-v4b/out" "itn-v3 itn jobs/itn-v2/out"; do set -- $pair; V=$1; K=$2; S=$3; O=jobs/$V/out; mkdir -p $O
  [ -e $O/model ] || ln -s ../../nfc-v5/out/$V $O/model
  cp $N/$K-test-prog.jsonl $O/test-prog.jsonl 2>/dev/null || cp $N/$K-prog.jsonl $O/test-prog.jsonl
  cp $N/$K-pred-prog.jsonl $O/pred-prog.jsonl
  nfc $S/test-multi.jsonl $O/test-multi.jsonl
  . envs/pii/bin/activate; [ -s $O/pred-multi-$V.jsonl ] || python src/tn/predict_tn.py $O/model $O/test-multi.jsonl $O/pred-multi-$V.jsonl; deactivate
  bash jobs/onnx-tn/run.sh $V
done
