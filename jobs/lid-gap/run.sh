#!/usr/bin/env bash
# lid-gap: do open LIDs handle romanized Indian languages? ~50 chat messages per (lang, script) cell, 10 Indic langs
# x {native, roman} + en = 21 cells (~1,050 short generations, TEST teacher Qwen3.8-27B, ~15 min). Score fastText-LID-218
# and GlotLID (fastText, envs/lid).
set -euo pipefail
cd ~/agent/rnd
O=jobs/lid-gap/out; mkdir -p $O
echo "[lid-gap] start $(date -Is)"
. envs/pii/bin/activate
for r in facebook/fasttext-language-identification cis-lmu/glotlid; do
  n=${r#*/}; [ -s ~/Downloads/model/$n/model.bin ] || python -c "from huggingface_hub import hf_hub_download as d; d('$r','model.bin',local_dir='$HOME/Downloads/model/$n')"
done
if [ ! -s $O/test.jsonl ]; then
  docker rm -f pii-teacher >/dev/null 2>&1 || true
  docker run -d --name pii-teacher --gpus all -v $HOME/Downloads/model:/models -p 127.0.0.1:8080:8080 \
    ghcr.io/ggml-org/llama.cpp:server-cuda13 \
    -m /models/Qwen3.8-27B-UD-Q4_K_XL-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf \
    -ngl 99 -c 32768 -np 8 --jinja --host 0.0.0.0 --port 8080 >/dev/null
  trap 'docker rm -f pii-teacher >/dev/null 2>&1 || true' EXIT
  for i in $(seq 1 120); do curl -sf 127.0.0.1:8080/health >/dev/null && break; sleep 5; done
  python src/lid/make_test.py $O/test.jsonl 50
  docker rm -f pii-teacher >/dev/null 2>&1 || true
fi
deactivate; . envs/lid/bin/activate
python src/lid/eval_lid.py $O/test.jsonl
echo "[lid-gap] end $(date -Is)"
