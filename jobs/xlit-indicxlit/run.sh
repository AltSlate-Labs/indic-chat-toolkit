#!/usr/bin/env bash
# IndicXlit baseline in a python:3.10 container (fairseq 0.12.2 needs it); nothing installed into our venvs.
set -euo pipefail
cd ~/agent/rnd
M=~/Downloads/model/IndicXlit/en-indic; O=jobs/xlit-indicxlit/out; mkdir -p $O envs/docker-pip-cache
[ -d $M ] || (mkdir -p $M && cd $M && python3 -c "import zipfile;zipfile.ZipFile('../indicxlit-en-indic-v1.0.zip').extractall('.')")
ls $M
docker run --rm -v $HOME/agent/rnd:/rnd -v $M:/model python:3.10 bash -c '
  set -e; export PIP_DEFAULT_TIMEOUT=180 PIP_RETRIES=10 PIP_NO_CACHE_DIR=1; pip -q install "pip<24.1" && pip -q install "numpy<2" torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
  && pip -q install cython "setuptools<70" wheel && pip install -q --no-build-isolation "fairseq @ git+https://github.com/facebookresearch/fairseq.git@v0.12.2" 2>&1 | grep -v -i "warning\|notice" ; python -c "import fairseq" && cd /rnd/src/xlit/indicxlit \
  && for T in A B; do python run_indicxlit.py /model /rnd/jobs/xlit-gap/out/test$T.jsonl /rnd/jobs/xlit-indicxlit/out/pred-$T.jsonl 2>&1 | grep -v "^[SWHDP]-" | tail -2; done'
. envs/pii/bin/activate
for T in A B; do echo "== IndicXlit (no rescoring) test $T"; python src/xlit/metrics.py jobs/xlit-gap/out/test$T.jsonl $O/pred-$T.jsonl; done
echo "[xlit-indicxlit] end $(date -Is)"
