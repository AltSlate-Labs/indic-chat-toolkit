"""Export a packaged sentence-transformers student (package_st.py output) to ONNX as one graph:
inputs input_ids, attention_mask -> outputs last_hidden_state, sentence_embedding (mean pool -> Dense 384->1024 -> L2).
Writes ST_DIR/onnx/model.onnx (fp32) and model_fp16.onnx (fp16 weights, fp32 inputs/outputs), then checks both against PyTorch on
st_check.txt + 300 dev inputs. python export_onnx.py ST_DIR [DEV_JSONL]   (envs/st, CPU)"""
import json, os, sys, numpy as np, torch, onnxruntime as ort
from transformers import AutoTokenizer, AutoModel
from onnxruntime.transformers.float16 import convert_float_to_float16
import onnx
st = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else None; os.makedirs(f"{st}/onnx", exist_ok=True)
tok = AutoTokenizer.from_pretrained(st); enc = AutoModel.from_pretrained(st, attn_implementation="eager").eval()
from safetensors.torch import load_file
w = load_file(f"{st}/2_Dense/model.safetensors"); dense = torch.nn.Linear(w["linear.weight"].shape[1], w["linear.weight"].shape[0])
dense.weight.data, dense.bias.data = w["linear.weight"], w["linear.bias"]
class Full(torch.nn.Module):
    def __init__(s): super().__init__(); s.enc, s.dense = enc, dense
    def forward(s, input_ids, attention_mask):
        h = s.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype); e = s.dense((h * m).sum(1) / m.sum(1).clamp(min=1e-9))
        return h, torch.nn.functional.normalize(e, dim=-1)
full = Full().eval(); b = tok(["kal meeting cancel ho gayi", "a longer second sentence for the dummy batch"], padding=True, return_tensors="pt")
f32, q8 = f"{st}/onnx/model.onnx", f"{st}/onnx/model_fp16.onnx"
torch.onnx.export(full, (b["input_ids"], b["attention_mask"]), f32, input_names=["input_ids", "attention_mask"],
                  output_names=["last_hidden_state", "sentence_embedding"], opset_version=17, dynamo=False,
                  dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                                "last_hidden_state": {0: "batch", 1: "seq"}, "sentence_embedding": {0: "batch"}})
# dynamic int8 (any variant: per-channel/tensor, MatMul-only) gave mean cos .94 vs fp32 on this model -> fp16 instead (.99995)
onnx.save(convert_float_to_float16(onnx.load(f32), keep_io_types=True), q8)
here = os.path.dirname(os.path.abspath(__file__))
S = [l.strip() for l in open(f"{here}/st_check.txt", encoding="utf-8") if l.strip()]
if dev: S += [json.loads(l)["input"] for l in list(open(dev, encoding="utf-8"))[:300]]
for path in (f32, q8):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"]); cs = []
    for i in range(0, len(S), 16):
        x = tok(S[i:i + 16], padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad(): ref = full(x["input_ids"], x["attention_mask"])[1].numpy()
        o = sess.run(["sentence_embedding"], {"input_ids": x["input_ids"].numpy(), "attention_mask": x["attention_mask"].numpy()})[0]
        cs += list((o * ref).sum(1))
    print(os.path.basename(path), f"{os.path.getsize(path) / 1e6:.0f} MB", "cos vs torch: min", round(float(min(cs)), 4), "mean", round(float(np.mean(cs)), 5), flush=True)
