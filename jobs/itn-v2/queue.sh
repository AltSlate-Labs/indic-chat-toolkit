#!/usr/bin/env bash
# Wait until no non-rnd containers run and the GPU is idle (2 checks 60 s apart), then run itn-v2 and the judge-based eval.
cd ~/agent/rnd
echo "[queue] waiting for the GPU $(date -Is)"
free_now() { ! docker ps --format '{{.Names}}' | grep -qv '^rnd-\|^pii-teacher$' && [ "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)" -lt 10 ]; }
until free_now && sleep 60 && free_now; do sleep 120; done
echo "[queue] GPU free $(date -Is)"
bash jobs/itn-v2/run.sh 2>&1 | tee jobs/itn-v2/log.txt
bash jobs/itn-eval/run.sh itn-v2 2>&1 | tee jobs/itn-eval/log-itn-v2.txt
