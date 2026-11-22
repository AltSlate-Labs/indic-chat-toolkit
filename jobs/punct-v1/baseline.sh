#!/usr/bin/env bash
# CPU baseline for punct-v1: 1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase on the three tests.
set -euo pipefail
cd ~/agent/rnd; O=jobs/punct-v1/out; . envs/pii/bin/activate
for T in chat ochat wiki; do CUDA_VISIBLE_DEVICES="" python src/punct/baseline_1800.py $O/test-$T.jsonl $O/base-$T.jsonl; echo "baseline $T done $(date -Is)"; done
