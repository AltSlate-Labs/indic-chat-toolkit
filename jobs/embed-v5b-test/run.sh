#!/usr/bin/env bash
# One-time TEST-split evaluation of embed-v5b (chosen on dev: sts-en 86.3, rest = v3), plus near-duplicate check of STSb test
# against the SNLI training text. Baselines' test numbers are in results/embed-test.txt.
set -euo pipefail
cd ~/agent/rnd; PY=envs/pii/bin/python
echo "[embed-v5b-test] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
$PY src/embed/evaluate.py test embed-v5b=jobs/embed-v5b/out:mean_head 2>&1 | grep -v -i "warn\|Loading\|deprecated"
echo "== neardup (STSb test vs SNLI train text)"
$PY src/embed/neardup_sts.py test jobs/embed-data-nli2/out/train.jsonl jobs/embed-v3/out jobs/embed-v5b/out 2>&1 | grep -v -i "warn\|Loading"
echo "[embed-v5b-test] end $(date -Is)"
