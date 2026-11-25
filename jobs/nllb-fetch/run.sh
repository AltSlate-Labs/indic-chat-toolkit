#!/usr/bin/env bash
# CPU/network only: stream the first 400 MB (highest LASER scores) of each allenai/nllb en-xx file, filter to 120k pairs/lang.
set -uo pipefail
cd ~/agent/rnd; O=~/Downloads/data/nllb-enxx; mkdir -p $O; B=https://storage.googleapis.com/allennlp-data-bucket/nllb
for p in hi:eng_Latn-hin_Deva:1 bn:ben_Beng-eng_Latn:0 ta:eng_Latn-tam_Taml:1 te:eng_Latn-tel_Telu:1 mr:eng_Latn-mar_Deva:1 \
         gu:eng_Latn-guj_Gujr:1 kn:eng_Latn-kan_Knda:1 ml:eng_Latn-mal_Mlym:1 pa:eng_Latn-pan_Guru:1 ur:eng_Latn-urd_Arab:1; do
  IFS=: read l f ef <<< "$p"
  [ -s $O/$l.tsv ] && continue
  curl -s $B/$f.gz | head -c 400000000 | zcat 2>/dev/null | envs/pii/bin/python src/embed/filter_nllb.py $l $ef 120000 $O/$l.tsv
done
echo "[nllb-fetch] end $(date -Is)"; du -sh $O
