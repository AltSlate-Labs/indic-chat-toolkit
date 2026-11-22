"""Predict word labels. python predict_wordlid.py MODEL_DIR TEST.jsonl OUT.jsonl [MAP]
TEST rows have "words"; OUT rows {"pred": [label per word]}. MAP=comi maps labels to hi / en / other (everything else -> other)."""
import json, sys, torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
d, test, out = sys.argv[1:4]; mp = sys.argv[4] if len(sys.argv) > 4 else ""
tok = AutoTokenizer.from_pretrained(d); m = AutoModelForTokenClassification.from_pretrained(d).eval().to("cuda" if torch.cuda.is_available() else "cpu")
I2L = m.config.id2label
rows = [json.loads(l)["words"] for l in open(test, encoding="utf-8")]; res = [None] * len(rows)
idx = sorted(range(len(rows)), key=lambda i: len(rows[i]))
for k in range(0, len(idx), 32):
    bi = idx[k:k + 32]; chunks = []  # long rows are cut into 100-word windows
    for i in bi: chunks += [(i, j, rows[i][j:j + 100]) for j in range(0, max(1, len(rows[i])), 100)]
    enc = tok([c[2] for c in chunks], is_split_into_words=True, truncation=True, max_length=512, padding=True, return_tensors="pt").to(m.device)
    with torch.no_grad(): pr = m(**enc).logits.argmax(-1).tolist()
    for b, (i, j, w) in enumerate(chunks):
        got, prev = {}, None
        for p, wid in enumerate(enc.word_ids(b)):
            if wid is not None and wid != prev: got[wid] = I2L[pr[b][p]]
            prev = wid
        if res[i] is None: res[i] = [None] * len(rows[i])
        for t in range(len(w)): res[i][j + t] = got.get(t, "other")
with open(out, "w", encoding="utf-8") as f:
    for p in res:
        if mp == "comi": p = [x if x in ("hi", "en") else "other" for x in p]
        f.write(json.dumps({"pred": p}) + "\n")
