#!/bin/bash
# Ornith-1.5-35B-A3B (training teacher, MIT) translations: (1) PHINC dev to measure the teacher (chrF++, never trained on);
# (2) Hinglish training sources (data0 e2h + PHINC train) -> English targets for mt-v2. Max 3.5 h of generation; resumable.
cd ~/agent/rnd; PY=envs/pii/bin/python; O=jobs/mt-orn1/out; mkdir -p $O
echo "[mt-orn1] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-' && { echo "other containers running; stop"; exit 1; } || true
[ -s $O/src.jsonl ] || $PY -c "
import json,random
R=[json.loads(l) for l in open('jobs/mt-data0/out/train.jsonl')]
seen=set();rows=[]
for r in R:
    if r['src'] in seen: continue
    seen.add(r['src']); rows.append({'src':r['src'],'kind':r['kind']})
random.Random(17).shuffle(rows)
rows.sort(key=lambda r: r['kind']!='phinc')   # PHINC train + human e2h first, then synthetic e2h
rows.sort(key=lambda r: r['kind']=='e2h-0')
with open('$O/src.jsonl','w') as f:
    for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
print('sources',len(rows))"
up() {
docker rm -f rnd-ornith 2>/dev/null || true
docker run -d --name rnd-ornith --network host --ipc host --gpus all --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $HOME/Downloads/model:/models:ro --entrypoint vllm vllm/vllm-openai:v0.32.0 serve /models/Ornith-1.5-35B-A3B-NVFP4 \
  --served-model-name ornith --host 127.0.0.1 --port 8011 --trust-remote-code --max-num-seqs 64 \
  --gpu-memory-utilization 0.45 --max-model-len 4096 --enable-prefix-caching --reasoning-parser qwen3 >/dev/null
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8011/health >/dev/null && break; sleep 15; done
curl -sf http://127.0.0.1:8011/health >/dev/null || { echo "vllm not healthy"; exit 1; }
echo "[mt-orn1] server up $(date -Is)"
}
trap 'docker logs --tail 30 rnd-ornith > '$O'/vllm-tail.txt 2>&1; docker rm -f rnd-ornith >/dev/null' EXIT
up || exit 1
[ -s $O/pred-ornith-phinc-dev.jsonl ] || { $PY src/mt/gen_teacher_mt.py http://127.0.0.1:8011 ornith jobs/mt-base/out/dev.jsonl $O/pred-ornith-phinc-dev.jsonl; }
T0=$(date +%s)
for try in 1 2 3 4; do   # vLLM hung once (18:58, all requests stalled) -> restart the server and resume; 210 min total
  LEFT=$(( 210 - ($(date +%s) - T0) / 60 )); [ $LEFT -le 0 ] && break
  $PY src/mt/gen_teacher_mt.py http://127.0.0.1:8011 ornith $O/src.jsonl $O/train-ornith.jsonl 1000000 $LEFT && break
  echo "[mt-orn1] generation stopped (try $try), restarting server $(date -Is)"; docker logs --tail 30 rnd-ornith > $O/vllm-hang$try.txt 2>&1; up || exit 1
done
echo "[mt-orn1] end $(date -Is)"
