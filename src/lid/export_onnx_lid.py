"""Export the indic-lid sequence classifier to ONNX (fp32 + per-channel dynamic int8, transformers.js layout onnx/) and
check label agreement with PyTorch on test B and test A cut to 5 words (agreement only; nothing is tuned on it).
python export_onnx_lid.py MODEL_DIR OUT_DIR A.jsonl B.jsonl"""
import json, os, sys, numpy as np, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

src, out, A, B = sys.argv[1:5]
os.makedirs(out + "/onnx", exist_ok=True)
tok = AutoTokenizer.from_pretrained(src)
model = AutoModelForSequenceClassification.from_pretrained(src, attn_implementation="eager").eval()
enc = tok(["kal mere interview tha", "ami tomake bhalobashi"], padding=True, return_tensors="pt")
f32, q8 = out + "/onnx/model.onnx", out + "/onnx/model_quantized.onnx"
torch.onnx.export(model, (enc["input_ids"], enc["attention_mask"]), f32, input_names=["input_ids", "attention_mask"],
                  output_names=["logits"], opset_version=17, dynamo=False,
                  dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch"}})
quantize_dynamic(f32, q8, weight_type=QuantType.QInt8, per_channel=True)
print("sizes MB:", round(os.path.getsize(f32) / 1e6), round(os.path.getsize(q8) / 1e6), flush=True)
texts = [json.loads(l)["text"] for l in open(B, encoding="utf-8")]
texts += [" ".join(json.loads(l)["text"].split()[:5]) for l in open(A, encoding="utf-8")][::4]
sess = {k: ort.InferenceSession(p, providers=["CPUExecutionProvider"]) for k, p in (("fp32", f32), ("int8", q8))}
agree = {k: 0 for k in sess}; maxdiff = 0.0
for i in range(0, len(texts), 64):
    e = tok(texts[i:i + 64], return_tensors="pt", truncation=True, max_length=64, padding=True)
    with torch.no_grad(): ref = model(**e).logits.numpy()
    feeds = {"input_ids": e["input_ids"].numpy(), "attention_mask": e["attention_mask"].numpy()}
    for k, s in sess.items():
        o = s.run(["logits"], feeds)[0]
        if k == "fp32": maxdiff = max(maxdiff, float(np.abs(o - ref).max()))
        agree[k] += int((o.argmax(-1) == ref.argmax(-1)).sum())
print("n", len(texts), "fp32 max |logit diff|", round(maxdiff, 5), {k: round(v / len(texts), 5) for k, v in agree.items()})
