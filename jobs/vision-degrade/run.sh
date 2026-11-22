#!/usr/bin/env bash
# vision-degrade: how much do JPEG re-compression and downscaling (as when screenshots are forwarded) hurt the en/hi redactor?
# Same 300 test screenshots as vision-probe; 2 recognizers + indic-pii-redactor-small (the Space's default mode). CPU only.
set -euo pipefail
cd ~/agent/rnd; M=~/Downloads/model; O=jobs/vision-degrade/out; S=jobs/vision-probe/out/test300
REC=$M/PP-OCRv5_mobile_rec_onnx,$M/devanagari_PP-OCRv5_mobile_rec_onnx
echo "[vision-degrade] start $(date -Is)"
for v in "1.0 60" "0.75 60" "0.5 60"; do set -- $v; d=$O/s$1-q$2
  envs/vision/bin/python src/vision/degrade.py $S $d $1 $2
  (cd src/vision && ../../envs/vision/bin/python ocr_shots.py $M/PP-OCRv5_mobile_det_onnx $REC ../../$d 2>&1 | grep -v "onnxruntime:")
  echo "== scale $1, JPEG q$2 (clean: latin .962 native .898 false blur .016)"
  (cd src && JOIN=" " ../envs/pii/bin/python vision/eval_shots.py ../$d altslate/indic-pii-redactor-small 2>&1 | grep -v -i "warn\|Loading" | head -4)
done
echo "[vision-degrade] end $(date -Is)"
