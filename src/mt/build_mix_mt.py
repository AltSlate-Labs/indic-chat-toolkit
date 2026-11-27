"""MT data round 1: Hinglish data0 (x UP) + allenai/nllb en-xx pairs from the embedder data (already filtered against FLORES/STS
eval sets): native Indic -> English ('xl') and romanized Indic -> English ('xl-roman'), N per language each; 300 xl-roman rows per
language held out as a multi-language dev. python build_mix_mt.py DATA0 EMBED_XL_JSONL OUT N UP"""
import json, os, random, sys
d0, xlf, out, N, UP = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]); os.makedirs(out, exist_ok=True)
rows = [json.loads(l) for l in open(f"{d0}/train.jsonl", encoding="utf-8")] * UP
pools = {}
for l in open(xlf, encoding="utf-8"):
    r = json.loads(l)
    if r["kind"] in ("xl", "xl-roman"): pools.setdefault((r["kind"], r["lang"]), []).append(r)
rng = random.Random(5); dev = []
for (k, lg), v in sorted(pools.items()):
    rng.shuffle(v)
    if k == "xl-roman": dev += [{"src": r["input"], "ref": r["target"], "lang": lg} for r in v[:300]]; v = v[300:]
    rows += [{"src": r["input"], "tgt": r["target"], "kind": f"{k}-{lg}"} for r in v[:N]]
rng.shuffle(rows)
with open(f"{out}/train.jsonl", "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(f"{out}/dev-xl.jsonl", "w", encoding="utf-8") as f:
    for r in dev: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("train", len(rows), "dev-xl", len(dev))
