#!/bin/bash
# ONE-TIME held-out test for the release candidate mt-v3b (chosen on dev: judge 3.55, FLORES te native fixed). Never tune after this.
# PHINC test (3,000) and COMI-LINGUA MT_test (5,000 sentences x roman/mixed) for copy, opus-mt-mul-en, nllb-600M (CC-BY-NC, eval only),
# mt-v3b; then the Qwen3.8-27B judge on 300 rows of each test set (human refs included as a system).
cd ~/agent/rnd; PY=envs/pii/bin/python; M=~/Downloads/model; O=jobs/mt-test/out; B=jobs/mt-base/out; C=$O/comi-test.jsonl
echo "[mt-test] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
mkdir -p $O; $PY src/mt/make_comi_mt.py test $C 5000
for spec in "copy copy x" "opus-mt-mul-en marian $M/opus-mt-mul-en" "nllb-600M nllb $M/nllb-200-distilled-600M" "mt-v3b m2m jobs/mt-v3b/out/model"; do
  set -- $spec
  $PY src/mt/eval_mt.py $B/test.jsonl test-$1-phinc $2 $3 2>&1 | grep "chrF"
  $PY src/mt/eval_mt.py $C test-$1-comi $2 $3 2>&1 | grep "chrF"
done
$PY -c "
import json
for a, b in [('$B/test.jsonl', '$O/pred-ref-phinc.jsonl'), ('$C', '$O/pred-ref-comi.jsonl')]:
    with open(b, 'w') as f:
        for l in open(a): r = json.loads(l); f.write(json.dumps({'src': r['src'], 'hyp': r['ref'], 'ref': r['ref']}, ensure_ascii=False) + '\n')"
docker rm -f pii-teacher >/dev/null 2>&1 || true
docker run -d --name pii-teacher --gpus all -v $HOME/Downloads/model:/models -p 127.0.0.1:8080:8080 \
  ghcr.io/ggml-org/llama.cpp:server-cuda13 -m /models/Qwen3.8-27B-UD-Q4_K_XL-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -ngl 99 -c 32768 -np 8 --jinja --host 0.0.0.0 --port 8080 >/dev/null
trap 'docker rm -f pii-teacher >/dev/null 2>&1 || true' EXIT
for i in $(seq 1 120); do curl -sf 127.0.0.1:8080/health >/dev/null && break; sleep 5; done
for t in phinc comi; do
  echo "== judge $t test"
  $PY src/mt/judge_mt.py 300 $O/judge-$t-test.json human-ref=$O/pred-ref-$t.jsonl opus-mt-mul-en=$B/pred-test-opus-mt-mul-en-$t.jsonl \
    nllb-600M=$B/pred-test-nllb-600M-$t.jsonl mt-v3b=$B/pred-test-mt-v3b-$t.jsonl 2>&1 | grep "judge\|Error"
done
echo "[mt-test] end $(date -Is)"
