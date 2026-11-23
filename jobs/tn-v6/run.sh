#!/usr/bin/env bash
# tn-v6: fixes from the tn-v5 judge audit (jobs/tn-v5-audit): unit names CLDR lacks (ur km/m/cm/l/hour/minute, bn/gu/ml gaps, mm
# everywhere); irregular ordinals written with the regular or English suffix (1ਵੀਂ, 1ویں, 1st, 3rd); kn/ml native zero used as anusvara
# inside a word (ರೆ೦ಟೆ). Data rebuilt (NFC), warm start tn-v5, 0.3 epoch, lr 1e-4, checkpoints every 500 steps.
# Note: the probe rows come from test A/B sentences we inspected in the audit, so the probe is diagnostic only; A/B judge scores are
# reported but these fixes were chosen by looking at A/B failures.
set -euo pipefail
cd ~/agent/rnd; O=jobs/tn-v6/out; G=jobs/tn-gap/out; V5=jobs/nfc-v5/out/tn-v5; mkdir -p $O
echo "[tn-v6] start $(date -Is)"
docker ps --format '{{.Names}}' | grep -v '^rnd-\|^pii-teacher$' && { echo "other containers running; stop"; exit 1; } || true
[ -s $O/train.jsonl ] || (cd src/tn && ../../envs/tn/bin/python fill_data.py ../../jobs/tn-lex/out/carriers.jsonl ../../$O 3)
envs/tn/bin/python - <<PY
import json, unicodedata as u
A={" ".join(json.loads(l)["text"].split()) for X in "AB" for l in open(f"$G/test{X}.jsonl",encoding="utf-8")}
for name in ("train","dev"):
    rows=[json.loads(l) for l in open(f"$O/{name}.jsonl",encoding="utf-8")]
    keep=[x for x in rows if " ".join(x["src"].split(": ",1)[1].split()) not in A]
    print(name, "dropped (exact match with tn-gap A/B):", len(rows)-len(keep), "non-NFC:", sum(x["tgt"]!=u.normalize("NFC",x["tgt"]) for x in keep))
    with open(f"$O/{name}.jsonl","w",encoding="utf-8") as f:
        for x in keep: f.write(json.dumps(x,ensure_ascii=False)+"\n")
import re
tr=[json.loads(l) for l in open("$O/train.jsonl",encoding="utf-8")]
print("train rows with mm:", sum(" mm" in x["src"] or re.search(r"\dmm",x["src"]) is not None for x in tr),
      "| ur km:", sum(x["lang"]=="ur" and "km" in x["src"] for x in tr),
      "| ordinal 1st/2nd/3rd in native text:", sum(re.search(r"\d(st|nd|rd)\b",x["src"]) is not None and x["lang"]!="ur" for x in tr),
      "| kn/ml zero-as-letter:", sum(x["lang"] in ("kn","ml") and ("ಂ" in x["tgt"] and "೦" in x["src"]) for x in tr))
PY
. envs/pii/bin/activate
[ -s $O/model/train_meta.json ] || (cd src/xlit && GPU_MEM_FRAC=0.6 python train_xlit.py --model ../../$V5 --train ../../$O/train.jsonl \
   --dev ../../$O/dev.jsonl --out ../../$O/model --epochs 0.3 --bs 32 --lr 1e-4 --max_src 768 --max_tgt 1536 --ckpt_every 500)
for M in v5 v6; do D=$([ $M = v5 ] && echo $V5 || echo $O/model)
  [ -s $O/pred-prog-$M.jsonl ] || python src/tn/predict_tn.py $D $O/test-prog.jsonl $O/pred-prog-$M.jsonl
  [ -s $O/pred-probe-$M.jsonl ] || python src/tn/predict_tn.py $D jobs/tn-v6/probe.jsonl $O/pred-probe-$M.jsonl; done
for X in A B; do [ -s $O/pred-$X.jsonl ] || python src/tn/predict_tn.py $O/model $G/test$X.jsonl $O/pred-$X.jsonl; done
[ -s $O/pred-sanity.jsonl ] || python src/tn/predict_tn.py $O/model src/tn/sanity.jsonl $O/pred-sanity.jsonl
deactivate
echo "== test-prog (new held-out carriers, v6 generator)"
envs/tn/bin/python src/tn/score_prog.py $O/test-prog.jsonl tn-v5=$O/pred-prog-v5.jsonl tn-v6=$O/pred-prog-v6.jsonl
echo "== probe (audited failures; diagnostic)"
paste -d'\t' jobs/tn-v6/probe.jsonl $O/pred-probe-v5.jsonl $O/pred-probe-v6.jsonl | python3 -c "
import sys,json
for l in sys.stdin:
    a,b,c=[json.loads(x) for x in l.rstrip('\n').split('\t')]; print(a['lang'], '|', a['text'][:70]); print('  v5:', b['pred'][:150]); print('  v6:', c['pred'][:150])"
docker rm -f pii-teacher >/dev/null 2>&1 || true
docker run -d --name pii-teacher --gpus all -v $HOME/Downloads/model:/models -p 127.0.0.1:8080:8080 \
  ghcr.io/ggml-org/llama.cpp:server-cuda13 -m /models/Qwen3.8-27B-UD-Q4_K_XL-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -ngl 99 -c 32768 -np 8 --jinja --host 0.0.0.0 --port 8080 >/dev/null
trap 'docker rm -f pii-teacher >/dev/null 2>&1 || true' EXIT
for i in $(seq 1 120); do curl -sf 127.0.0.1:8080/health >/dev/null && break; sleep 5; done
for X in A B; do echo "== test $X judged (tn-v5: A .797 / B .758)"; envs/tn/bin/python src/tn/judge.py $G/test$X.jsonl tn-v6=$O/pred-$X.jsonl; done
echo "[tn-v6] end $(date -Is)"
