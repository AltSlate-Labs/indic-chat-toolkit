#!/usr/bin/env bash
# Release prep for tn-v6: multi/pair tests (v5 vs v6), then jobs/onnx-tn/run.sh tn-v6 (model-fix, ONNX, int8, ORT checks).
set -euo pipefail
cd ~/agent/rnd; O=jobs/tn-v6/out; V5=jobs/nfc-v5/out/tn-v5
cp $O/pred-prog-v6.jsonl $O/pred-prog.jsonl
. envs/pii/bin/activate
for T in multi pair; do for M in v5 v6; do D=$([ $M = v5 ] && echo $V5 || echo $O/model)
  [ -s $O/pred-$T-tn-$M.jsonl ] || python src/tn/predict_tn.py $D $O/test-$T.jsonl $O/pred-$T-tn-$M.jsonl; done; done
deactivate
for T in multi pair; do echo "== test-$T"; envs/tn/bin/python src/tn/score_prog.py $O/test-$T.jsonl tn-v5=$O/pred-$T-tn-v5.jsonl tn-v6=$O/pred-$T-tn-v6.jsonl | grep exact; done
bash jobs/onnx-tn/run.sh tn-v6
