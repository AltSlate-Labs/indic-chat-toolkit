#!/usr/bin/env bash
# ONNX export of an indic-lid model: bash run.sh v4|v5
set -euo pipefail
cd ~/agent/rnd/src/lid && . ../../envs/pii/bin/activate
V=$1; O=../../jobs/onnx-lid/out/$V; mkdir -p $O
python export_onnx_lid.py ../../jobs/lid-$V/out/model $O ../../jobs/lid-gap/out/dakshina-test.jsonl ../../jobs/lid-testb/out/testb.jsonl
ls -la $O/onnx; echo "[onnx-lid $V] end $(date -Is)"
