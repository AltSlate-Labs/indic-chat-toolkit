#!/usr/bin/env bash
# nfc-probe: indic-numtowords gives Telugu (and some kn/ml/ta) words in decomposed Unicode (ై = U+0C46 U+0C56, not U+0C48), so the
# released TN model writes non-NFC text and the ITN model was trained on non-NFC input. Measure both:
#  (1) TN v4b: judge test A/B again with outputs NFC-normalized (vs .720/.703; te .32/.32)
#  (2) ITN v2: test-A-itn with the spoken input NFC-normalized, as real users type it (vs the original, partly non-NFC input)
set -euo pipefail
cd ~/agent/rnd; O=jobs/nfc-probe/out; mkdir -p $O; E=jobs/itn-eval/out
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
nfc() { python3 -c "
import json,sys,unicodedata as u
for l in open(sys.argv[1],encoding='utf-8'):
    x=json.loads(l); print(json.dumps({k:(u.normalize('NFC',v) if isinstance(v,str) else v) for k,v in x.items()},ensure_ascii=False))" $1 > $2; }
echo "[nfc-probe] start $(date -Is)"
nfc $E/test-A-itn.jsonl $O/test-A-itn-nfc.jsonl
. envs/pii/bin/activate; python src/tn/predict_tn.py jobs/itn-v2/out/model $O/test-A-itn-nfc.jsonl $O/itn-pred-A-nfc.jsonl; deactivate
echo "== ITN test-A-itn: original input vs NFC input (refs are NFC-insensitive digits)"
python3 src/itn/score_itn.py $E/test-A-itn.jsonl itn-v2-orig=$E/itn-v2/pred-A.jsonl
python3 src/itn/score_itn.py $O/test-A-itn-nfc.jsonl itn-v2-nfc-input=$O/itn-pred-A-nfc.jsonl
for X in A B; do nfc jobs/tn-v4b/out/pred-$X.jsonl $O/tn-pred-$X-nfc.jsonl; done
docker rm -f pii-teacher >/dev/null 2>&1 || true
docker run -d --name pii-teacher --gpus all -v $HOME/Downloads/model:/models -p 127.0.0.1:8080:8080 \
  ghcr.io/ggml-org/llama.cpp:server-cuda13 -m /models/Qwen3.8-27B-UD-Q4_K_XL-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -ngl 99 -c 32768 -np 8 --jinja --host 0.0.0.0 --port 8080 >/dev/null
trap 'docker rm -f pii-teacher >/dev/null 2>&1 || true' EXIT
for i in $(seq 1 120); do curl -sf 127.0.0.1:8080/health >/dev/null && break; sleep 5; done
for X in A B; do echo "== TN test $X, outputs NFC-normalized (v4b raw: A .720 te .32 / B .703 te .32)"
  envs/tn/bin/python src/tn/judge.py jobs/tn-gap/out/test$X.jsonl tn-v4b-nfc=$O/tn-pred-$X-nfc.jsonl; done
echo "[nfc-probe] end $(date -Is)"
