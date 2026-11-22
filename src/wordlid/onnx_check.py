"""Word-level accuracy of exported ONNX files vs PyTorch predictions. python onnx_check.py MODEL_DIR ONNX_DIR TEST.jsonl TORCH_PRED.jsonl N [comi]
TEST rows: {"words", "labels"} (test-mix) or {"words", "gold"} (COMI; labels mapped to hi/en/other)."""
import json, sys, numpy as np, onnxruntime as ort
from transformers import AutoTokenizer, AutoConfig
md, od, test, ref, n = sys.argv[1:6]; comi = len(sys.argv) > 6
tok = AutoTokenizer.from_pretrained(md); I2L = AutoConfig.from_pretrained(md).id2label
rows = [json.loads(l) for l in open(test, encoding="utf-8")][:int(n)]; tp = [json.loads(l)["pred"] for l in open(ref)][:int(n)]
gold = lambda x: x["gold"] if comi else x["labels"]
for name in ("model.onnx", "model_quantized.onnx"):
    s = ort.InferenceSession(f"{od}/{name}", providers=["CPUExecutionProvider"]); ok = tot = same = 0
    for x, t in zip(rows, tp):
        w = x["words"][:100]; e = tok(w, is_split_into_words=True, truncation=True, max_length=512, return_tensors="np")
        lg = s.run(["logits"], {"input_ids": e["input_ids"].astype(np.int64), "attention_mask": e["attention_mask"].astype(np.int64)})[0][0]
        got, prev = {}, None
        for p, wid in enumerate(e.word_ids()):
            if wid is not None and wid != prev: got[wid] = I2L[int(lg[p].argmax())]
            prev = wid
        P = [got.get(i, "other") for i in range(len(w))]
        if comi: P = [q if q in ("hi", "en") else "other" for q in P]
        for g, q, r in zip(gold(x)[:len(w)], P, t[:len(w)]):
            if comi and g not in ("hi", "en"): continue
            tot += 1; ok += g == q; same += q == r
    print(f"{name:22s} word acc {ok/tot:.4f}  same as torch {same/tot:.4f}  (n={tot})", flush=True)
