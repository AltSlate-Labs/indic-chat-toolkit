"""Direct onnxruntime greedy decoding of an exported TN model, compared with PyTorch predictions.
Step 1 runs decoder_model (returns the KV cache), later steps run decoder_with_past_model, as transformers.js does with the merged
decoder (the no-past loop was O(n^2) and took >30 min for 150 TN rows).
python onnx_greedy_tn.py ONNX_DIR TEST.jsonl TORCH_PRED.jsonl N SUFFIX   (rows have "src" and "tgt"; SUFFIX "" for fp32 or "_quantized")"""
import json, sys, time, numpy as np, onnxruntime as ort
d, test_f, ref_f, n, suf = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), (sys.argv[5] if len(sys.argv) > 5 else "")
S = lambda f: ort.InferenceSession(f"{d}/{f}{suf}.onnx")
enc, dec0, decp = S("encoder_model"), S("decoder_model"), S("decoder_with_past_model")
names0 = [o.name for o in dec0.get_outputs()]; namesp = [o.name for o in decp.get_outputs()]
test = [json.loads(l) for l in open(test_f, encoding="utf-8")][:n]
torch_pred = [json.loads(l)["pred"] for l in open(ref_f, encoding="utf-8")][:n]; pred = []; t0 = time.time()
N = lambda s: " ".join(s.split())
for x in test:
    ids = np.array([[b + 3 for b in x["src"].encode("utf-8")] + [1]], dtype=np.int64); am = np.ones_like(ids)
    h = enc.run(None, {"input_ids": ids, "attention_mask": am})[0]
    o = dict(zip(names0, dec0.run(None, {"input_ids": np.array([[0]], dtype=np.int64), "encoder_hidden_states": h, "encoder_attention_mask": am})))
    past = {k.replace("present", "past_key_values"): v for k, v in o.items() if k != "logits"}; out = [0]
    for _ in range(4 * ids.shape[1] + 64):
        t = int(o["logits"][0, -1].argmax()); out.append(t)
        if t == 1: break
        o = dict(zip(namesp, decp.run(None, {"input_ids": np.array([[t]], dtype=np.int64), "encoder_attention_mask": am, **past})))
        past.update({k.replace("present", "past_key_values"): v for k, v in o.items() if k != "logits"})  # encoder KV stays from step 1
    pred.append(bytes(b - 3 for b in out[1:] if b >= 3).decode("utf-8", errors="ignore"))
ex = lambda P: sum(N(p) == N(x["tgt"]) for p, x in zip(P, test)) / len(test)
print(suf or "fp32", f"exact {ex(pred):.3f} torch {ex(torch_pred):.3f} identical-to-torch {sum(a == b for a, b in zip(pred, torch_pred)) / n:.3f}",
      f"{(time.time() - t0) / n * 1000:.0f} ms/row", flush=True)
