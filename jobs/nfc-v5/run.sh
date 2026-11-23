#!/usr/bin/env bash
# nfc-v5: retrain on Unicode-NFC data (indic-numtowords writes Telugu ై decomposed; see jobs/nfc-probe: TN te outputs non-NFC,
# ITN te on NFC input .38 vs .95). Same rows as tn-v4b / itn-v2, NFC-normalized; warm starts.
#  tn-v5 : tn-v4b + 0.2 epoch, lr 1e-4.   itn-v3: itn-v2 + 0.5 epoch, lr 1e-4. Tests NFC-normalized too.
set -euo pipefail
cd ~/agent/rnd; O=jobs/nfc-v5/out; mkdir -p $O; T=jobs/tn-v4b/out; I=jobs/itn-v2/out; E=jobs/itn-eval/out; Z=jobs/itn-eval/zeros; G=jobs/tn-gap/out
echo "[nfc-v5] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
nfc() { python3 -c "
import json,sys,unicodedata as u
for l in open(sys.argv[1],encoding='utf-8'):
    x=json.loads(l); print(json.dumps({k:(u.normalize('NFC',v) if isinstance(v,str) else v) for k,v in x.items()},ensure_ascii=False))" $1 > $2; }
for f in train dev test-prog; do nfc $T/$f.jsonl $O/tn-$f.jsonl; done
nfc $I/train.jsonl $O/itn-train.jsonl; nfc jobs/itn-v1/out/dev.jsonl $O/itn-dev.jsonl
nfc jobs/itn-v1/out/test-prog.jsonl $O/itn-prog.jsonl; nfc $Z/zeros-test.jsonl $O/itn-zeros.jsonl; nfc $E/test-A-itn.jsonl $O/itn-A.jsonl
. envs/pii/bin/activate
[ -s $O/tn-v5/train_meta.json ] || (cd src/xlit && GPU_MEM_FRAC=0.6 python train_xlit.py --model ../../$T/model --train ../../$O/tn-train.jsonl \
   --dev ../../$O/tn-dev.jsonl --out ../../$O/tn-v5 --epochs 0.2 --bs 32 --lr 1e-4 --max_src 768 --max_tgt 1536 --ckpt_every 1000)
[ -s $O/itn-v3/train_meta.json ] || (cd src/xlit && GPU_MEM_FRAC=0.6 python train_xlit.py --model ../../$I/model --train ../../$O/itn-train.jsonl \
   --dev ../../$O/itn-dev.jsonl --out ../../$O/itn-v3 --epochs 0.5 --bs 32 --lr 1e-4 --max_src 768 --max_tgt 1536 --ckpt_every 1000)
[ -s $O/tn-pred-prog.jsonl ] || python src/tn/predict_tn.py $O/tn-v5 $O/tn-test-prog.jsonl $O/tn-pred-prog.jsonl
for X in A B; do [ -s $O/tn-pred-$X.jsonl ] || python src/tn/predict_tn.py $O/tn-v5 $G/test$X.jsonl $O/tn-pred-$X.jsonl; done
for X in prog zeros A; do [ -s $O/itn-pred-$X.jsonl ] || python src/tn/predict_tn.py $O/itn-v3 $O/itn-$X.jsonl $O/itn-pred-$X.jsonl; done
for X in prog zeros; do [ -s $O/itn-v2-pred-$X.jsonl ] || python src/tn/predict_tn.py $I/model $O/itn-$X.jsonl $O/itn-v2-pred-$X.jsonl; done
[ -s $O/tn-pred-sanity.jsonl ] || python src/tn/predict_tn.py $O/tn-v5 src/tn/sanity.jsonl $O/tn-pred-sanity.jsonl
deactivate
python3 -c "
import json,unicodedata as u,collections
c=collections.Counter()
for l,t in zip(open('$O/tn-pred-A.jsonl'),open('$G/testA.jsonl')):
    p=json.loads(l)['pred']; c[json.loads(t)['lang']]+=p!=u.normalize('NFC',p)
print('tn-v5 test A non-NFC outputs by lang:', dict(c))"
echo "== TN test-prog (NFC)"; envs/tn/bin/python src/tn/score_prog.py $O/tn-test-prog.jsonl tn-v5=$O/tn-pred-prog.jsonl
echo "== ITN NFC inputs: itn-v2 vs itn-v3"
python3 src/itn/score_itn.py $O/itn-A.jsonl itn-v2=jobs/nfc-probe/out/itn-pred-A-nfc.jsonl itn-v3=$O/itn-pred-A.jsonl
python3 src/itn/score_itn.py $O/itn-prog.jsonl itn-v2=$O/itn-v2-pred-prog.jsonl itn-v3=$O/itn-pred-prog.jsonl
python3 src/itn/score_itn.py $O/itn-zeros.jsonl itn-v2=$O/itn-v2-pred-zeros.jsonl itn-v3=$O/itn-pred-zeros.jsonl
docker rm -f pii-teacher >/dev/null 2>&1 || true
docker run -d --name pii-teacher --gpus all -v $HOME/Downloads/model:/models -p 127.0.0.1:8080:8080 \
  ghcr.io/ggml-org/llama.cpp:server-cuda13 -m /models/Qwen3.8-27B-UD-Q4_K_XL-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -ngl 99 -c 32768 -np 8 --jinja --host 0.0.0.0 --port 8080 >/dev/null
trap 'docker rm -f pii-teacher >/dev/null 2>&1 || true' EXIT
for i in $(seq 1 120); do curl -sf 127.0.0.1:8080/health >/dev/null && break; sleep 5; done
for X in A B; do echo "== TN test $X judged (tn-v4b with NFC'd outputs: A .798 / B .770)"
  envs/tn/bin/python src/tn/judge.py $G/test$X.jsonl tn-v5=$O/tn-pred-$X.jsonl; done
paste -d'\t' src/tn/sanity.jsonl $O/tn-pred-sanity.jsonl | grep '"te"'
echo "[nfc-v5] end $(date -Is)"
