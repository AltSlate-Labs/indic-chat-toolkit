"""Export the cross-encoder to ONNX (input_ids, attention_mask -> logits) as fp32, fp16 and dynamic int8 (per-channel), then check
each against PyTorch on 15 MIRACLReranking dev queries x 100 candidates (hi/bn/te, native and romanized query): max |score diff|,
and nDCG@10 computed with each runtime. python export_onnx.py DIR   (envs/st, CPU)"""
import os, sys, numpy as np, torch, onnx, onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.transformers.float16 import convert_float_to_float16
d = sys.argv[1]; os.makedirs(f"{d}/onnx", exist_ok=True); os.environ["RERANK_N"] = "15"
tok = AutoTokenizer.from_pretrained(d); m = AutoModelForSequenceClassification.from_pretrained(d, attn_implementation="eager").eval()
b = tok(["q one", "query two"], ["a passage", "another longer passage here"], padding=True, return_tensors="pt")
f32, f16, q8 = f"{d}/onnx/model.onnx", f"{d}/onnx/model_fp16.onnx", f"{d}/onnx/model_quantized.onnx"
torch.onnx.export(m, (b["input_ids"], b["attention_mask"]), f32, input_names=["input_ids", "attention_mask"], output_names=["logits"],
                  opset_version=17, dynamo=False, dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "logits": {0: "b"}})
onnx.save(convert_float_to_float16(onnx.load(f32), keep_io_types=True), f16)
quantize_dynamic(f32, q8, weight_type=QuantType.QInt8, per_channel=True)
sys.argv = ["x"]; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_rerank as E   # loads the dev data (RERANK_N=40 queries per task)
def torch_score(q, ps):
    out = []
    with torch.no_grad():
        for i in range(0, len(ps), 25):
            x = tok([q] * len(ps[i:i + 25]), ps[i:i + 25], padding=True, truncation="only_second", max_length=384, return_tensors="pt")
            out += m(**x).logits[:, 0].tolist()
    return out
def onnx_scorer(path):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    def f(q, ps):
        out = []
        for i in range(0, len(ps), 25):
            x = tok([q] * len(ps[i:i + 25]), ps[i:i + 25], padding=True, truncation="only_second", max_length=384, return_tensors="np")
            out += s.run(["logits"], {"input_ids": x["input_ids"].astype(np.int64), "attention_mask": x["attention_mask"].astype(np.int64)})[0][:, 0].tolist()
        return out
    return f
tasks = ["hi", "bn", "te", "hi-ro", "bn-ro", "te-ro"]
ref = {t: [torch_score(q, ps) for q, ps, g in E.DATA[t]] for t in tasks}
def nd(scores, t): return np.mean([x for x in (E.ndcg10(s, g) for s, (q, ps, g) in zip(scores, E.DATA[t])) if x is not None])
print("torch  ", " ".join(f"{t}:{nd(ref[t], t):.3f}" for t in tasks), flush=True)
for name, path in [("fp32", f32), ("fp16", f16), ("int8", q8)]:
    f = onnx_scorer(path); sc = {t: [f(q, ps) for q, ps, g in E.DATA[t]] for t in tasks}
    diff = max(np.max(np.abs(np.array(a) - np.array(b))) for t in tasks for a, b in zip(sc[t], ref[t]))
    print(f"{name:7s}", f"{os.path.getsize(path) / 1e6:.0f}MB maxdiff {diff:.3f}", " ".join(f"{t}:{nd(sc[t], t):.3f}" for t in tasks), flush=True)
