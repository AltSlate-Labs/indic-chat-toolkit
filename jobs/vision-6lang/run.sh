#!/usr/bin/env bash
# vision-6lang: screenshot PII redaction beyond Hindi. Recognizers: PP-OCRv5 mobile (zh/Latin) + devanagari + ta + te (all ONNX,
# Apache-2.0; Paddle has no ONNX bn/gu/kn/ml/pa recognizer). PII model: altslate/indic-pii-redactor-base.
# (a) 300 screenshots from jobs/v7/out/test7-final.jsonl (6 languages, held-out names); (b) the en/hi set again with 4 recognizers
# and the small model, to check that extra recognizers do not hurt.
set -euo pipefail
cd ~/agent/rnd; M=~/Downloads/model; O=jobs/vision-6lang/out
REC=$M/PP-OCRv5_mobile_rec_onnx,$M/devanagari_PP-OCRv5_mobile_rec_onnx,$M/ta_PP-OCRv5_mobile_rec_onnx,$M/te_PP-OCRv5_mobile_rec_onnx
echo "[vision-6lang] start $(date -Is)"
envs/vision/bin/python src/vision/render_shots.py jobs/v7/out/test7-final.jsonl $O/test7 300 11
(cd src/vision && ../../envs/vision/bin/python ocr_shots.py $M/PP-OCRv5_mobile_det_onnx $REC ../../$O/test7 2>&1 | grep -v "onnxruntime:")
echo "== test7 (6 languages), base model"
(cd src && JOIN=" " ../envs/pii/bin/python vision/eval_shots.py ../$O/test7 altslate/indic-pii-redactor-base 2>&1 | grep -v -i "warn\|Loading")
mkdir -p $O/test300-4rec && cp jobs/vision-probe/out/test300/*.png jobs/vision-probe/out/test300/shots.jsonl $O/test300-4rec/
(cd src/vision && ../../envs/vision/bin/python ocr_shots.py $M/PP-OCRv5_mobile_det_onnx $REC ../../$O/test300-4rec 2>&1 | grep -v "onnxruntime:")
echo "== en/hi set with 4 recognizers, small model (2 recognizers gave latin .962 native .898 false blur .016)"
(cd src && JOIN=" " ../envs/pii/bin/python vision/eval_shots.py ../$O/test300-4rec altslate/indic-pii-redactor-small 2>&1 | grep -v -i "warn\|Loading" | head -4)
echo "[vision-6lang] end $(date -Is)"
