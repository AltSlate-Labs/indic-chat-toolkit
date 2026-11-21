"""Direct onnxruntime greedy decoding (no optimum): encoder_model + decoder_model (no past), to verify exported graphs.
python onnx_greedy.py ONNX_DIR TEST.jsonl TORCH_PRED.jsonl N SUFFIX   (SUFFIX "" for fp32 or "_quantized")"""
import json, os, sys, time, numpy as np, onnxruntime as ort
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import score
d, test_f, ref_f, n, suf = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), (sys.argv[5] if len(sys.argv) > 5 else "")
enc = ort.InferenceSession(f"{d}/encoder_model{suf}.onnx"); dec = ort.InferenceSession(f"{d}/decoder_model{suf}.onnx")
print([i.name for i in enc.get_inputs()], [i.name for i in dec.get_inputs()])
def bytes_ids(s): return [b + 3 for b in s.encode("utf-8")] + [1]  # ByT5: byte+3, eos=1
test = [json.loads(l) for l in open(test_f, encoding="utf-8")][:n]
torch_pred = [json.loads(l)["pred"] for l in open(ref_f, encoding="utf-8")][:n]; pred = []; t0 = time.time()
for x in test:
    ids = np.array([bytes_ids(f'{x["lang"]}: {x["roman"]}')], dtype=np.int64); am = np.ones_like(ids)
    h = enc.run(None, {"input_ids": ids, "attention_mask": am})[0]; out = [0]
    for _ in range(4 * ids.shape[1] + 32):
        lg = dec.run(None, {"input_ids": np.array([out], dtype=np.int64), "encoder_hidden_states": h, "encoder_attention_mask": am})[0]
        t = int(lg[0, -1].argmax()); out.append(t)
        if t == 1: break
    pred.append(bytes(b - 3 for b in out[1:] if b >= 3).decode("utf-8", errors="ignore"))
print(suf or "fp32", "CER", round(score(test, pred)[0], 4), "torch", round(score(test, torch_pred)[0], 4),
      "identical", round(sum(a == b for a, b in zip(pred, torch_pred)) / n, 3), f"{(time.time() - t0) / n * 1000:.0f} ms/msg")
print(pred[:3], torch_pred[:3])
