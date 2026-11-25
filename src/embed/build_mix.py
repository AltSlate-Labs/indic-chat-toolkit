"""embed-base data: one mixed epoch drawn from the three small-model rounds, with their teacher vectors merged (no re-encoding).
python build_mix.py OUT SRC:KIND[,KIND]:N ...   e.g. jobs/embed-data/out:*:1300000 jobs/embed-data-v2/out:native2:600000
Each SRC is a data dir with train.jsonl + targets.json + targets.f16.npy; N random rows of the listed kinds (* = all kinds).
OUT gets train.jsonl, dev.jsonl (from the first SRC), targets.json and targets.f16.npy covering every target used."""
import json, os, random, shutil, sys, numpy as np
out = sys.argv[1]; specs = sys.argv[2:]; os.makedirs(out, exist_ok=True); r = random.Random(0)
rows, need = [], {}
for sp in specs:
    src, kinds, n = sp.split(":"); n = int(n); ks = None if kinds == "*" else set(kinds.split(","))
    cand = [l for l in open(f"{src}/train.jsonl", encoding="utf-8") if ks is None or json.loads(l)["kind"] in ks]
    pick = [json.loads(l) for l in r.sample(cand, min(n, len(cand)))]; del cand
    rows += pick; need.setdefault(src, set()).update(x["target"] for x in pick)
    print(sp, "->", len(pick), flush=True)
first = specs[0].split(":")[0]
dev = [json.loads(l) for l in open(f"{first}/dev.jsonl", encoding="utf-8")]; need.setdefault(first, set()).update(x["target"] for x in dev)
texts, where = [], {}
for src, ts in need.items():
    for t in ts:
        if t not in where: where[t] = (src, None); texts.append(t)
pos = {t: i for i, t in enumerate(texts)}
M = np.lib.format.open_memmap(f"{out}/targets.f16.npy", mode="w+", dtype=np.float16, shape=(len(texts), 1024))
for src in need:
    st = json.load(open(f"{src}/targets.json", encoding="utf-8")); T = np.load(f"{src}/targets.f16.npy", mmap_mode="r")
    idx = [(pos[t], i) for i, t in enumerate(st) if t in where and where[t][0] == src]
    if idx:
        a, b = zip(*sorted(idx, key=lambda x: x[1])); M[list(a)] = T[list(b)]
    print(src, "vectors", len(idx), flush=True)
M.flush(); json.dump(texts, open(f"{out}/targets.json", "w", encoding="utf-8"), ensure_ascii=False)
r.shuffle(rows)
with open(f"{out}/train.jsonl", "w", encoding="utf-8") as f:
    for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")
shutil.copy(f"{first}/dev.jsonl", f"{out}/dev.jsonl")
chk = np.load(f"{out}/targets.f16.npy", mmap_mode="r"); z = int((np.abs(chk[:: max(1, len(texts) // 20000)]).sum(1) == 0).sum())
print("train", len(rows), "targets", len(texts), "zero-vector rows in sample", z)
