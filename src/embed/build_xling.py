"""embed-v3 data: cross-lingual distillation (Reimers & Gurevych 2020 style) from NLLB en-xx pairs (filter_nllb.py output).
python build_xling.py PREV_DATA NLLB_DIR OUT [REPLAY] [ROMAN_FRAC]
- xl      : Indic sentence -> teacher(English translation)
- xl-roman: the same Indic sentence romanized by romanizer.py (seed 3) -> teacher(English), for ROMAN_FRAC of the pairs
- en      : English side -> teacher(English)
- replay  : REPLAY random rows of PREV_DATA/train.jsonl (native/roman/mixed/en/hinglish/native2)
dev.jsonl = PREV_DATA/dev.jsonl."""
import json, os, random, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from romanizer import Romanizer
prev, nd, out = sys.argv[1:4]; REPLAY = int(sys.argv[4]) if len(sys.argv) > 4 else 600000; RF = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
os.makedirs(out, exist_ok=True); r = random.Random(0); R = Romanizer(seed=3)
old = [l for l in open(f"{prev}/train.jsonl", encoding="utf-8")]
rows = [json.loads(l) for l in r.sample(old, min(REPLAY, len(old)))]; del old
for l in ["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "ur"]:
    n = 0
    for line in open(f"{nd}/{l}.tsv", encoding="utf-8"):
        e, x = line.split("\t")[:2]
        rows += [{"input": x, "target": e, "kind": "xl", "lang": l}, {"input": e, "target": e, "kind": "en", "lang": "en"}]
        if r.random() < RF: rows.append({"input": R.sentence(x, l), "target": e, "kind": "xl-roman", "lang": l})
        n += 1
    print(l, n, flush=True)
r.shuffle(rows)
with open(f"{out}/train.jsonl", "w", encoding="utf-8") as f:
    for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")
shutil.copy(f"{prev}/dev.jsonl", f"{out}/dev.jsonl"); print("train", len(rows))
