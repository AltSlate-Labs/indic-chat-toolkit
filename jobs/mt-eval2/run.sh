#!/bin/bash
# NLLB-600M (eval only, CC-BY-NC) on FLORES dev with correct source codes; then Qwen3.8-27B judge (test judge) on 300 PHINC dev rows.
cd ~/agent/rnd; PY=envs/pii/bin/python; M=~/Downloads/model; O=jobs/mt-eval2/out; B=jobs/mt-base/out
echo "[mt-eval2] start $(date -Is)"
while tmux has -t job-mt-eval1 2>/dev/null; do sleep 30; done
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
mkdir -p $O
$PY src/mt/eval_mt.py jobs/mt-eval1/out/flores-dev.jsonl nllb-600M-flores nllb $M/nllb-200-distilled-600M 2>&1 | grep "chrF"
$PY -c "
import json
with open('$O/pred-ref.jsonl','w') as f:
    for l in open('$B/dev.jsonl'): r=json.loads(l); f.write(json.dumps({'src':r['src'],'hyp':r['ref'],'ref':r['ref']},ensure_ascii=False)+'\n')"
docker rm -f pii-teacher >/dev/null 2>&1 || true
docker run -d --name pii-teacher --gpus all -v $HOME/Downloads/model:/models -p 127.0.0.1:8080:8080 \
  ghcr.io/ggml-org/llama.cpp:server-cuda13 -m /models/Qwen3.8-27B-UD-Q4_K_XL-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -ngl 99 -c 32768 -np 8 --jinja --host 0.0.0.0 --port 8080 >/dev/null
trap 'docker rm -f pii-teacher >/dev/null 2>&1 || true' EXIT
for i in $(seq 1 120); do curl -sf 127.0.0.1:8080/health >/dev/null && break; sleep 5; done
$PY src/mt/judge_mt.py 300 $O/judge-phinc-dev.json human-ref=$O/pred-ref.jsonl copy=$B/pred-copy.jsonl \
  opus-mt-mul-en=$B/pred-opus-mt-mul-en.jsonl nllb-600M=$B/pred-nllb-600M.jsonl mt-v1=jobs/mt-v1/out/pred-dev.jsonl \
  qwen27b=$B/pred-qwen27b-phinc.jsonl 2>&1 | grep "judge\|Error"
echo "[mt-eval2] end $(date -Is)"
