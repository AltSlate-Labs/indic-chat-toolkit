"""xlit-v2 training data = xlit-v1 train.jsonl plus Aksharantar (ai4bharat, train split only; CC-BY manual / CC0 mined):
 (a) word pairs: mined IndicCorp pairs with score >= that language's median, plus all Samanantar pairs (names, loanwords);
     N_BIG words for kn/ml/te/ta (where IndicXlit beat us), N_SMALL for the rest; Dakshina-sourced pairs skipped (already in v1);
 (b) pseudo-sentences of 10-25 random Aksharantar words (N_PSEUDO per language) so that long outputs are seen in training (v1 loops).
Sinhala is not in Aksharantar. Dev = v1 dev (unchanged). python build_data_v2.py V1_DIR OUT_DIR"""
import json, os, random, sys, zipfile
A = os.path.expanduser("~/Downloads/data/aksharantar/")
ISO = {"bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan", "ml": "mal", "mr": "mar", "pa": "pan", "sd": "sid", "ta": "tam", "te": "tel", "ur": "urd"}
BIG = {"kn", "ml", "te", "ta"}
N_BIG, N_SMALL, N_PSEUDO = (int(os.environ.get(k, d)) for k, d in (("N_BIG", 60000), ("N_SMALL", 25000), ("N_PSEUDO", 2000)))
v1, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True); r = random.Random(12)
train = [l for l in open(f"{v1}/train.jsonl", encoding="utf-8")]; stats = {}


def row(lang, roman, native): return json.dumps({"src": f"{lang}: {roman}", "tgt": native, "lang": lang}, ensure_ascii=False) + "\n"


for l, iso in ISO.items():
    mined, other = [], []
    with zipfile.ZipFile(f"{A}{iso}.zip").open(f"{iso}_train.json") as f:
        for line in f:
            x = json.loads(line); ro, na = x["english word"].strip(), x["native word"].strip()
            if not ro or not na or " " in ro or " " in na or not ro.isascii(): continue
            if x["source"] == "IndicCorp" and x["score"] is not None: mined.append((float(x["score"]), ro, na))
            elif x["source"] in ("Samanantar", "Wikidata"): other.append((ro, na))
    mined.sort(reverse=True); good = [(ro, na) for _, ro, na in mined[:len(mined) // 2]]
    n = N_BIG if l in BIG else N_SMALL
    words = other[:n // 4] + r.sample(good, min(len(good), n - min(len(other), n // 4)))
    train += [row(l, ro, na) for ro, na in words]
    for _ in range(N_PSEUDO):
        ws = r.sample(words, r.randint(10, 25)); ro, na = " ".join(w for w, _ in ws), " ".join(w for _, w in ws)
        if len(ro.encode()) <= 360 and len(na.encode()) <= 720: train.append(row(l, ro, na))  # train_xlit truncates at 384 / 768 bytes
    stats[l] = (len(mined), len(other), len(words))
r.shuffle(train)
open(f"{out}/train.jsonl", "w", encoding="utf-8").writelines(train)
open(f"{out}/dev.jsonl", "w", encoding="utf-8").writelines(open(f"{v1}/dev.jsonl", encoding="utf-8"))
print("train", len(train), "per lang (mined, names, used):", stats)
