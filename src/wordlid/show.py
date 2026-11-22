"""Print word labels + message summary for hand-written lines (dev eyeballing only). python show.py MODEL_DIR LINES.txt"""
import collections, sys, torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1]); m = AutoModelForTokenClassification.from_pretrained(sys.argv[1]).eval()
for s in open(sys.argv[2], encoding="utf-8").read().splitlines():
    w = s.split(); e = tok(w, is_split_into_words=True, return_tensors="pt")
    with torch.no_grad(): p = m(**e).logits[0].argmax(-1).tolist()
    lab = {}
    for i, wid in enumerate(e.word_ids()):
        if wid is not None and wid not in lab: lab[wid] = m.config.id2label[p[i]]
    L = [lab[i] for i in range(len(w))]; c = collections.Counter(x for x in L if x != "other"); n = sum(c.values()) or 1
    print(" ".join(f"{a}/{b}" for a, b in zip(w, L)), "|", ", ".join(f"{k} {round(100 * v / n)}%" for k, v in c.most_common()))
