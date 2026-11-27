#!/bin/bash
# Chat-style test for languages other than Hindi (PHINC/COMI are Hindi only): xlit test B (696 casual chat messages written by
# Qwen3.8-27B in native script and romanized; never used for MT training), 10 MT languages (si, sd dropped), roman + native.
# No English references: translate with mt-v3b and nllb-600M (eval only), then the Qwen judge scores every row (source + candidate).
cd ~/agent/rnd; PY=envs/pii/bin/python; M=~/Downloads/model; O=jobs/mt-chat-test/out; B=jobs/mt-base/out; F=$O/chat-test.jsonl
echo "[mt-chat-test] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
mkdir -p $O
python3 -c "
import json
with open('$F', 'w', encoding='utf-8') as f:
    for l in open('jobs/xlit-gap/out/testB.jsonl', encoding='utf-8'):
        r = json.loads(l)
        if r['lang'] in ('si', 'sd'): continue
        for k in ('roman', 'native'): f.write(json.dumps({'src': r[k], 'ref': r[k], 'lang': r['lang'], 'kind': k}, ensure_ascii=False) + '\n')"
wc -l $F
$PY src/mt/eval_mt.py $F chat-mt-v3b m2m jobs/mt-v3b/out/model 2>&1 | grep -c chrF
$PY src/mt/eval_mt.py $F chat-nllb-600M nllb $M/nllb-200-distilled-600M 2>&1 | grep -c chrF
docker rm -f pii-teacher >/dev/null 2>&1 || true
docker run -d --name pii-teacher --gpus all -v $HOME/Downloads/model:/models -p 127.0.0.1:8080:8080 \
  ghcr.io/ggml-org/llama.cpp:server-cuda13 -m /models/Qwen3.8-27B-UD-Q4_K_XL-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -ngl 99 -c 32768 -np 8 --jinja --host 0.0.0.0 --port 8080 >/dev/null
trap 'docker rm -f pii-teacher >/dev/null 2>&1 || true' EXIT
for i in $(seq 1 120); do curl -sf 127.0.0.1:8080/health >/dev/null && break; sleep 5; done
N=$(wc -l < $F)
$PY src/mt/judge_mt.py $N $O/judge-chat.json mt-v3b=$B/pred-chat-mt-v3b.jsonl nllb-600M=$B/pred-chat-nllb-600M.jsonl 2>&1 | grep "judge\|Error"
python3 -c "
import json
rows = [json.loads(l) for l in open('$F', encoding='utf-8')]; J = json.load(open('$O/judge-chat.json'))
for k, sc in J['scores'].items():
    for kind in ('roman', 'native'):
        g = {}
        for i, s in zip(J['ix'], sc):
            if rows[i]['kind'] == kind: g.setdefault(rows[i]['lang'], []).append(s)
        print(f'{k:10} {kind:6}', ' '.join(f'{l}:{sum(v)/len(v):.2f}' for l, v in sorted(g.items())), '| mean %.2f' % (sum(sum(v) for v in g.values()) / sum(len(v) for v in g.values())))"
echo "[mt-chat-test] end $(date -Is)"
