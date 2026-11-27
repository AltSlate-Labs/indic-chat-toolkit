"""Split PHINC (human Hinglish -> English, CC-BY-4.0) into train/dev/test with a fixed seed. Drops rows whose source equals the
target after lower-casing (already-English rows), empty rows and duplicate sources. python split_phinc.py OUT_DIR"""
import json, os, random, sys, pandas as pd
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
d = pd.read_csv(os.path.expanduser("~/Downloads/data/phinc/PHINC.csv")).dropna()
norm = lambda t: " ".join(str(t).lower().split())
rows, seen = [], set()
for s, t in zip(d["Sentence"], d["English_Translation"]):
    s, t = " ".join(str(s).split()), " ".join(str(t).split())
    if not s or not t or norm(s) == norm(t) or norm(s) in seen: continue
    seen.add(norm(s)); rows.append({"src": s, "ref": t})
random.Random(13).shuffle(rows)
parts = {"test": rows[:3000], "dev": rows[3000:4000], "train": rows[4000:]}
for k, v in parts.items():
    with open(f"{out}/{k}.jsonl", "w", encoding="utf-8") as f:
        for r in v: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print({k: len(v) for k, v in parts.items()}, "dropped", len(d) - len(rows))
