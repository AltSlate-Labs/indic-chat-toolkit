"""itn-v2 extra data: numbers with internal zeros (1024, 4096, 1077, 20,05,003 ...). itn-v1 dropped the zero after "thousand"
(probe: 1024 -> 124 in ta/ml, 4096 -> 496 in te/ta/ml/bn; test A ml 1077 -> 177) because uniform random numbers rarely have them.
python zeros.py CARRIERS.jsonl OUT_DIR N_TRAIN_PER_LANG   (run in envs/tn)
Uses NUMBER / AMOUNT / QUANTITY carriers, split exactly as fill_data.py does (i % 10 == 0 -> test), so test rows use held-out carriers."""
import json, random, re, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tn"))
from fill_data import Lang, LANGS, good_carrier, W
cf, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3]); os.makedirs(out, exist_ok=True)
car = [c for c in (json.loads(x) for x in open(cf, encoding="utf-8")) if good_carrier(c)]
random.Random(0).shuffle(car)  # same order as fill_data.py
split = {"test": [c for i, c in enumerate(car) if i % 10 == 0], "train": [c for i, c in enumerate(car) if i % 10]}
def zval(r):  # a number with at least one internal zero digit
    while True:
        v = r.choice([r.randint(1000, 9999), r.randint(10000, 99999), r.randint(100000, 9999999), r.randint(100, 999)])
        s = str(v)
        if "0" in s[1:-1] or (len(s) >= 4 and s[1] == "0"): return v
langs = {l: Lang(l) for l in LANGS}
for name, k in (("train", n), ("test", 30)):
    r = random.Random(f"zeros-{name}"); rows = []
    for l, L in langs.items():
        cs = [c for c in split[name] if c["lang"] == l and c["kind"] in ("NUMBER", "AMOUNT")]
        for _ in range(k):
            c = r.choice(cs); v = zval(r); ph = f"⟦{c['kind']}⟧"
            if c["kind"] == "AMOUNT": sp, wr = f"{W(v, l)} {L.x['rupees']}", "₹" + L.igroup(v)
            else: sp, wr = W(v, l), str(v)
            src, tgt = c["carrier"].replace(ph, sp), c["carrier"].replace(ph, wr)
            if r.random() < 0.5: src, tgt = (" ".join(re.sub(r"(?<!\d)[,.।॥?!;:](?!\d)|۔|،|؟", " ", t).split()) for t in (src, tgt))
            rows.append({"src": f"{l}-itn: {src}", "tgt": tgt, "lang": l, "kind": "ZERO-" + c["kind"]})
    with open(f"{out}/zeros-{name}.jsonl", "w", encoding="utf-8") as f:
        for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(name, len(rows))
