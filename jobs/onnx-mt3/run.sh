#!/usr/bin/env bash
# CPU only: ONNX export of the MT release candidate mt-v3b (optimum, text2text-generation-with-past), then transformers.js-named
# variants: *_fp16.onnx (onnxconverter_common, fp32 IO kept) and *_quantized.onnx (ORT dynamic int8). Node check runs on the Mac.
set -euo pipefail
cd ~/agent/rnd; M=jobs/mt-v3b/out/model; O=jobs/onnx-mt/out/mt-v3b
echo "[onnx-mt3] start $(date -Is)"
mkdir -p $O; . envs/export/bin/activate
export CUDA_VISIBLE_DEVICES=""
[ -s $O/onnx/encoder_model.onnx ] || optimum-cli export onnx --model $M --task text2text-generation-with-past --device cpu $O/onnx 2>&1 | grep -v -i warn | tail -5
python - <<PY
import onnx, os
from onnxconverter_common import float16
from onnxruntime.quantization import quantize_dynamic, QuantType
d = "$O/onnx"
for n in ["encoder_model", "decoder_model_merged"]:
    f = f"{d}/{n}.onnx"
    if not os.path.exists(f"{d}/{n}_fp16.onnx"):
        m = float16.convert_float_to_float16(onnx.load(f), keep_io_types=True, disable_shape_infer=True)
        onnx.save(m, f"{d}/{n}_fp16.onnx")
    if not os.path.exists(f"{d}/{n}_quantized.onnx"):
        quantize_dynamic(f, f"{d}/{n}_quantized.onnx", weight_type=QuantType.QInt8)
    print(n, {s: round(os.path.getsize(f"{d}/{n}{s}.onnx") / 2**20) for s in ["", "_fp16", "_quantized"]}, "MB")
PY
rm -f $O/onnx/decoder_model.onnx $O/onnx/decoder_with_past_model.onnx
ls -la $O/onnx
echo "[onnx-mt3] end $(date -Is)"
