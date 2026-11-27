#!/bin/bash
# mt-v4: chat-style data for all 10 languages (mt-chat-test: mt-v3b judge roman non-hi 2.4-3.3, native 2.3-4.1; Tamil/Telugu negation lost).
# (1) Ornith (training teacher) translates ~24k Ornith-written chat messages (native side; LID/xlit projects; test-B overlaps dropped,
# 40/lang held out as chat dev); (2) continue mt-v3b 5k steps lr 3e-5 on data2 + chat pairs (roman + native) x4 + native te x2;
# (3) dev evals + Qwen judge (PHINC dev 300, chat dev 800) for mt-v3b vs mt-v4. ~2 h. Chat TEST (xlit test B) only after choosing.
cd ~/agent/rnd; PY=envs/pii/bin/python; O=jobs/mt-v4/out; C=jobs/mt-chat1/out; B=jobs/mt-base/out
echo "[mt-v4] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-' && { echo "other containers running; stop"; exit 1; } || true
mkdir -p $O $C
[ -s $C/src.jsonl ] || $PY src/mt/build_chat_mt.py src $C jobs/mt-chat-test/out/chat-test.jsonl jobs/lid-chat/out/chat-train.jsonl jobs/lid-chat4/out/chat-weak.jsonl
up() {
docker rm -f rnd-ornith 2>/dev/null || true
docker run -d --name rnd-ornith --network host --ipc host --gpus all --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $HOME/Downloads/model:/models:ro --entrypoint vllm vllm/vllm-openai:v0.32.0 serve /models/Ornith-1.5-35B-A3B-NVFP4 \
  --served-model-name ornith --host 127.0.0.1 --port 8011 --trust-remote-code --max-num-seqs 64 \
  --gpu-memory-utilization 0.45 --max-model-len 4096 --enable-prefix-caching --reasoning-parser qwen3 >/dev/null
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8011/health >/dev/null && break; sleep 15; done
curl -sf http://127.0.0.1:8011/health >/dev/null || { echo "vllm not healthy"; return 1; }
echo "[mt-v4] ornith up $(date -Is)"
}
if [ ! -s $O/train.jsonl ]; then
  up || exit 1
  for try in 1 2 3; do
    $PY src/mt/gen_teacher_mt.py http://127.0.0.1:8011 ornith $C/src.jsonl $C/train-ornith.jsonl 1000000 60 && break
    echo "[mt-v4] generation stopped (try $try), restarting $(date -Is)"; up || exit 1
  done
  docker rm -f rnd-ornith >/dev/null
  $PY src/mt/build_chat_mt.py mix $O jobs/mt-data2/out/train.jsonl $C/train-ornith.jsonl 4
fi
[ -s $O/train_meta.json ] || $PY src/mt/train_mt.py --base jobs/mt-v3b/out/model --src_lang auto --data $O/train.jsonl --dev $B/dev.jsonl \
  --out $O --steps 5000 --lr 3e-5 --ckpt_every 2500 2>&1 | grep --line-buffered -v -i warn | grep --line-buffered "params\|rows\|resumed\|step [0-9]*000/\|DEV\|Error\|error"
$PY src/mt/eval_mt.py jobs/mt-data1/out/dev-xl.jsonl mt-v4-xl m2m $O/model 2>&1 | grep chrF
$PY src/mt/eval_mt.py jobs/mt-eval1/out/flores-dev.jsonl mt-v4-flores m2m $O/model 2>&1 | grep chrF
$PY src/mt/eval_mt.py jobs/mt-eval3/out/comi-dev.jsonl mt-v4-comi m2m $O/model 2>&1 | grep chrF
cp $O/pred-dev.jsonl $B/pred-mt-v4-phinc.jsonl
for m in mt-v3b mt-v4; do D=jobs/$m/out/model; $PY src/mt/eval_mt.py $C/dev.jsonl chatdev-$m m2m $D 2>&1 | grep -c chrF; done
docker rm -f pii-teacher >/dev/null 2>&1 || true
docker run -d --name pii-teacher --gpus all -v $HOME/Downloads/model:/models -p 127.0.0.1:8080:8080 \
  ghcr.io/ggml-org/llama.cpp:server-cuda13 -m /models/Qwen3.8-27B-UD-Q4_K_XL-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -ngl 99 -c 32768 -np 8 --jinja --host 0.0.0.0 --port 8080 >/dev/null
trap 'docker rm -f pii-teacher rnd-ornith >/dev/null 2>&1 || true' EXIT
for i in $(seq 1 120); do curl -sf 127.0.0.1:8080/health >/dev/null && break; sleep 5; done
$PY src/mt/judge_mt.py 300 $O/judge-phinc-dev.json mt-v4=$B/pred-mt-v4-phinc.jsonl 2>&1 | grep "judge\|Error"
N=$(wc -l < $C/dev.jsonl)
$PY src/mt/judge_mt.py $N $O/judge-chatdev.json mt-v3b=$B/pred-chatdev-mt-v3b.jsonl mt-v4=$B/pred-chatdev-mt-v4.jsonl 2>&1 | grep "judge\|Error"
python3 -c "
import json
rows = [json.loads(l) for l in open('$C/dev.jsonl', encoding='utf-8')]; J = json.load(open('$O/judge-chatdev.json'))
for k, sc in J['scores'].items():
    for kind in ('roman', 'native'):
        g = {}
        for i, s in zip(J['ix'], sc):
            if rows[i]['kind'] == kind: g.setdefault(rows[i]['lang'], []).append(s)
        print(f'{k:8} {kind:6}', ' '.join(f'{l}:{sum(v)/len(v):.2f}' for l, v in sorted(g.items())), '| mean %.2f' % (sum(sum(v) for v in g.values()) / sum(len(v) for v in g.values())))"
rm -f $O/ckpt.pt $O/train.jsonl
echo "[mt-v4] end $(date -Is)"
