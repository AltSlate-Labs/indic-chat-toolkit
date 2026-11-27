#!/bin/bash
# ONE-TIME tests for mt-v4 (chosen on dev 06:31: chat dev judge 2.98 vs mt-v3b 2.23; PHINC dev judge 3.58 vs 3.55). Same files and
# judge rows (seed 3) as job-mt-test / job-mt-chat-test, so the numbers compare with mt-v3b, nllb-600M and opus-mt-mul-en there.
cd ~/agent/rnd; PY=envs/pii/bin/python; O=jobs/mt-test4/out; B=jobs/mt-base/out; T=jobs/mt-test/out; F=jobs/mt-chat-test/out/chat-test.jsonl
echo "[mt-test4] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
mkdir -p $O; D=jobs/mt-v4/out/model
$PY src/mt/eval_mt.py $B/test.jsonl test-mt-v4-phinc m2m $D 2>&1 | grep "chrF"
$PY src/mt/eval_mt.py $T/comi-test.jsonl test-mt-v4-comi m2m $D 2>&1 | grep "chrF"
$PY src/mt/eval_mt.py $F chat-mt-v4 m2m $D 2>&1 | grep -c "chrF"
docker rm -f pii-teacher >/dev/null 2>&1 || true
docker run -d --name pii-teacher --gpus all -v $HOME/Downloads/model:/models -p 127.0.0.1:8080:8080 \
  ghcr.io/ggml-org/llama.cpp:server-cuda13 -m /models/Qwen3.8-27B-UD-Q4_K_XL-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -ngl 99 -c 32768 -np 8 --jinja --host 0.0.0.0 --port 8080 >/dev/null
trap 'docker rm -f pii-teacher >/dev/null 2>&1 || true' EXIT
for i in $(seq 1 120); do curl -sf 127.0.0.1:8080/health >/dev/null && break; sleep 5; done
for t in phinc comi; do
  echo "== judge $t test"; $PY src/mt/judge_mt.py 300 $O/judge-$t-test.json mt-v4=$B/pred-test-mt-v4-$t.jsonl 2>&1 | grep "judge\|Error"
done
N=$(wc -l < $F)
$PY src/mt/judge_mt.py $N $O/judge-chat.json mt-v4=$B/pred-chat-mt-v4.jsonl 2>&1 | grep "judge\|Error"
python3 -c "
import json
rows = [json.loads(l) for l in open('$F', encoding='utf-8')]; J = json.load(open('$O/judge-chat.json'))
for k, sc in J['scores'].items():
    for kind in ('roman', 'native'):
        g = {}
        for i, s in zip(J['ix'], sc):
            if rows[i]['kind'] == kind: g.setdefault(rows[i]['lang'], []).append(s)
        print(f'{k:10} {kind:6}', ' '.join(f'{l}:{sum(v)/len(v):.2f}' for l, v in sorted(g.items())), '| mean %.2f' % (sum(sum(v) for v in g.values()) / sum(len(v) for v in g.values())))"
echo "[mt-test4] end $(date -Is)"
