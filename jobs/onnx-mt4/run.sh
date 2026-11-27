#!/usr/bin/env bash
# CPU only. ONNX for a trained M2M100 MT model (recipe from mt-v3b, see STATE Known-good): fp32 export; fp16 via ORT OnnxModel on the
# un-merged decoders then optimum merge_decoders; int8 via quantize_dynamic with EnableSubgraph. usage: bash jobs/onnx-mt4/run.sh mt-v4
set -euo pipefail
V=$1; cd ~/agent/rnd; M=jobs/$V/out/model; O=jobs/onnx-mt/out/$V; T=$O/tmp
echo "[onnx-mt4 $V] start $(date -Is)"
. envs/export/bin/activate; export CUDA_VISIBLE_DEVICES=""
mkdir -p $O/onnx
[ -s $O/onnx/encoder_model.onnx ] || optimum-cli export onnx --model $M --task text2text-generation-with-past --device cpu $O/onnx 2>&1 | grep -v -i warn | tail -2
[ -s $T/decoder_with_past_model.onnx ] || optimum-cli export onnx --model $M --task text2text-generation-with-past --device cpu --no-post-process $T 2>&1 | grep -v -i warn | tail -2
python - <<PY
import onnx, os, onnxruntime as ort
from onnxruntime.transformers.onnx_model import OnnxModel
from onnxruntime.quantization import quantize_dynamic, QuantType
from optimum.onnx import merge_decoders
d, t = "$O/onnx", "$T"
def fp16(src, dst):
    m = OnnxModel(onnx.load(src)); m.convert_float_to_float16(keep_io_types=True, use_symbolic_shape_infer=False); m.save_model_to_file(dst)
fp16(f"{d}/encoder_model.onnx", f"{d}/encoder_model_fp16.onnx")
for n in ["decoder_model", "decoder_with_past_model"]: fp16(f"{t}/{n}.onnx", f"{t}/{n}_fp16.onnx")
merge_decoders(f"{t}/decoder_model_fp16.onnx", f"{t}/decoder_with_past_model_fp16.onnx", save_path=f"{d}/decoder_model_merged_fp16.onnx", strict=False)
for n in ["encoder_model", "decoder_model_merged"]:
    quantize_dynamic(f"{d}/{n}.onnx", f"{d}/{n}_quantized.onnx", weight_type=QuantType.QInt8, extra_options={"EnableSubgraph": True})
for n in ["encoder_model", "decoder_model_merged"]:
    for s in ["", "_fp16", "_quantized"]:
        f = f"{d}/{n}{s}.onnx"; ort.InferenceSession(f, providers=["CPUExecutionProvider"]); print(n + s, round(os.path.getsize(f) / 2**20), "MB loads")
PY
rm -rf $T $O/onnx/decoder_model.onnx $O/onnx/decoder_with_past_model.onnx
echo "[onnx-mt4 $V] end $(date -Is)"
