"""Predict punctuation classes with an ONNX file (onnxruntime, CPU). python predict_onnx.py MODEL_DIR ONNX_FILE TEST.jsonl OUT.jsonl"""
import json, sys, numpy as np, onnxruntime as ort
from transformers import AutoTokenizer, AutoConfig
md, of, test, out = sys.argv[1:5]
tok = AutoTokenizer.from_pretrained(md); I2L = AutoConfig.from_pretrained(md).id2label
s = ort.InferenceSession(of, providers=["CPUExecutionProvider"])
with open(out, "w") as f:
    for l in open(test, encoding="utf-8"):
        w = json.loads(l)["words"]; P = []
        for j in range(0, len(w), 100):
            ch = w[j:j + 100]; e = tok(ch, is_split_into_words=True, truncation=True, max_length=512, return_tensors="np")
            lg = s.run(["logits"], {"input_ids": e["input_ids"].astype(np.int64), "attention_mask": e["attention_mask"].astype(np.int64)})[0][0]
            got, prev = {}, None
            for p, wid in enumerate(e.word_ids()):
                if wid is not None and wid != prev: got[wid] = I2L[int(lg[p].argmax())]
                prev = wid
            P += [got.get(i, "O") for i in range(len(ch))]
        f.write(json.dumps({"pred": P}) + "\n")
