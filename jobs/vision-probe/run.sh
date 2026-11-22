#!/usr/bin/env bash
# vision-probe: screenshot PII redaction, first measurement. 300 chat screenshots rendered from the en/hi/Hinglish PII test set
# (jobs/testgen-v0/out/test-final.jsonl; held out from PII training) -> PP-OCRv5 mobile (ONNX, own numpy/Pillow runner) ->
# altslate/indic-pii-redactor-small on the OCR text -> boxes; scored against the rendered gold boxes.
set -euo pipefail
cd ~/agent/rnd; O=jobs/vision-probe/out/test300; M=~/Downloads/model
echo "[vision-probe] start $(date -Is)"
envs/vision/bin/python src/vision/render_shots.py jobs/testgen-v0/out/test-final.jsonl $O 300 7
(cd src/vision && ../../envs/vision/bin/python ocr_shots.py $M/PP-OCRv5_mobile_det_onnx $M/PP-OCRv5_mobile_rec_onnx ../../$O 2>&1 | grep -v "onnxruntime:")
(cd src && ../envs/pii/bin/python vision/eval_shots.py ../$O altslate/indic-pii-redactor-small 2>&1 | grep -v -i "warn\|Loading")
echo "[vision-probe] end $(date -Is)"
