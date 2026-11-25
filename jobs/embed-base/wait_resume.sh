#!/usr/bin/env bash
# Run embed-base chunks until training is finished, each one only after the GPU has had no non-rnd containers for 10
# consecutive minutes (CLAUDE.md: CPU-only while others use the GPU). Checks every minute; gives up after 24 h.
cd ~/agent/rnd; free=0
for i in $(seq 1 1440); do
  [ -s jobs/embed-base/out/train_meta.json ] && { echo "[wait_resume] training finished $(date -Is)"; exit 0; }
  if docker ps --format "{{.Names}}" | grep -qv "^rnd-\|^pii-teacher$"; then free=0; else free=$((free+1)); fi
  if [ $free -ge 10 ]; then echo "[wait_resume] GPU free 10 min, running a chunk $(date -Is)"; bash jobs/embed-base/run.sh; free=0; fi
  sleep 60
done
echo "[wait_resume] gave up $(date -Is)"
