"""Try int8 variants of an fp32 ONNX: per-channel, and with the classifier MatMul kept in fp32.
python requant_onnx.py FP32.onnx OUTDIR"""
import sys, onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
f32, out = sys.argv[1:3]
g = onnx.load(f32, load_external_data=False).graph
mm = [n.name for n in g.node if n.op_type in ("MatMul", "Gemm")]
last = mm[-1:]  # classifier head
print("last matmul:", last, "of", len(mm))
quantize_dynamic(f32, f"{out}/pc.onnx", weight_type=QuantType.QInt8, per_channel=True)
quantize_dynamic(f32, f"{out}/pc_head.onnx", weight_type=QuantType.QInt8, per_channel=True, nodes_to_exclude=last)
quantize_dynamic(f32, f"{out}/head.onnx", weight_type=QuantType.QInt8, nodes_to_exclude=last)
