"""Score word-level LID predictions on the COMI test. python score_comi.py TEST.jsonl NAME=PRED.jsonl ...
PRED rows: {"pred": [label per word]} with labels mapped to hi / en / other (see comi_test.py for the metric)."""
import json, re, sys
test = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
LAT = re.compile(r"[A-Za-z]")
for arg in sys.argv[2:]:
    name, f = arg.split("=", 1); P = [json.loads(l)["pred"] for l in open(f, encoding="utf-8")]
    a = b = c = d = 0
    for x, p in zip(test, P):
        for w, g, q in zip(x["words"], x["gold"], p):
            if g not in ("hi", "en"): continue
            c += 1; d += q == g
            if LAT.search(w) and not re.search("[ऀ-ॿ]", w): a += 1; b += q == g
    print(f"{name:12s} latin hi/en acc {b/a:.3f} (n={a})   all hi/en acc {d/c:.3f} (n={c})")
