"""Baseline: 1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase via the `punctuators` package (ONNX).
python baseline_1800.py TEST.jsonl OUT.jsonl   Input = the test words joined with spaces (no punctuation); the output sentences are
joined and split on whitespace; if the word count changed, the row is marked {"skip": true}."""
import json, sys
from punctuators.models import PunctCapSegModelONNX
CLS = {",": "COMMA", "،": "COMMA", "।": "PERIOD", "॥": "PERIOD", ".": "PERIOD", "۔": "PERIOD", "!": "PERIOD", "?": "QUESTION", "؟": "QUESTION"}
m = PunctCapSegModelONNX.from_pretrained("1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase")
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
outs = []
B = 32
for i in range(0, len(rows), B):
    res = m.infer([" ".join(x["words"]) for x in rows[i:i + B]])
    for x, sents in zip(rows[i:i + B], res):
        ws = " ".join(sents).split()
        if len(ws) != len(x["words"]): outs.append({"skip": True}); continue
        outs.append({"pred": [next((CLS[c] for c in reversed(w) if c in CLS), "O") if w[-1] in CLS else "O" for w in ws]})
with open(sys.argv[2], "w") as f:
    for o in outs: f.write(json.dumps(o) + "\n")
