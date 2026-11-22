"""Export the token-classification model to ONNX (fp32 + dynamic int8) and check it against PyTorch.
python export_onnx.py MODEL_DIR OUT_DIR DEV.jsonl"""
import json, sys, os, numpy as np, torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

src, out, dev = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(src)
model = AutoModelForTokenClassification.from_pretrained(src, attn_implementation="eager").eval()
enc = tok(["Mera aadhaar 2345 6789 0123 hai", "PAN ABCPS1234K"], padding=True, return_tensors="pt")
f32 = f"{out}/model.onnx"
torch.onnx.export(model, (enc["input_ids"], enc["attention_mask"]), f32, input_names=["input_ids", "attention_mask"],
                  output_names=["logits"], opset_version=17, dynamo=False,
                  dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                                "logits": {0: "batch", 1: "seq"}})
q8 = f"{out}/model_quantized.onnx"
quantize_dynamic(f32, q8, weight_type=QuantType.QInt8)
print("sizes MB:", round(os.path.getsize(f32) / 1e6), round(os.path.getsize(q8) / 1e6), flush=True)

texts = [json.loads(l)["text"] for l in open(dev, encoding="utf-8")]
sess = {k: ort.InferenceSession(p, providers=["CPUExecutionProvider"]) for k, p in (("fp32", f32), ("int8", q8))}
agree = {k: [0, 0] for k in sess}; maxdiff = 0.0
for t in texts:
    e = tok(t, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        ref = model(**e).logits[0].numpy()
    feeds = {"input_ids": e["input_ids"].numpy(), "attention_mask": e["attention_mask"].numpy()}
    for k, s in sess.items():
        o = s.run(["logits"], feeds)[0][0]
        if k == "fp32": maxdiff = max(maxdiff, float(np.abs(o - ref).max()))
        agree[k][0] += int((o.argmax(-1) == ref.argmax(-1)).sum()); agree[k][1] += ref.shape[0]
print("fp32 max |logit diff| vs torch:", round(maxdiff, 5))
for k, (a, n) in agree.items():
    print(f"{k}: token label agreement with torch {a}/{n} = {a / n:.5f}")
