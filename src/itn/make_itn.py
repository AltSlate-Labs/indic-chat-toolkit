"""Build ITN (spoken -> written) data from a TN data dir made by src/tn/fill_data.py (rows carry "itn" = canonical written form).
python make_itn.py TN_DIR TESTB.jsonl OUT_DIR
 train/dev/test-prog/test-multi/test-pair: src = "<lang>-itn: <spoken>", tgt = canonical written. Sentence punctuation is stripped
 from ~50% of inputs (ASR models such as IndicConformer write none) and then also from the target; ',' '.' ':' between digits stay.
 test-B: tn-gap test B reversed: src = teacher spoken form, ref = the original written message (scored by digit sequence)."""
import json, random, re, sys, os
tn, testb, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
PUNCT = re.compile(r"(?<!\d)[,.।॥?!;:\"“”'‘’()\[\]⟨⟩«»-](?!\d)|۔|،|؟")
strip = lambda t: " ".join(PUNCT.sub(" ", t).split())
for name in ("train", "dev", "test-prog", "test-multi", "test-pair"):
    r = random.Random(name); rows = []
    for l in open(f"{tn}/{name}.jsonl", encoding="utf-8"):
        x = json.loads(l)
        if not x.get("itn"): continue
        sp, wr = x["tgt"], x["itn"]
        if r.random() < 0.5: sp, wr = strip(sp), strip(wr)
        rows.append({"src": f"{x['lang']}-itn: {sp}", "tgt": wr, "lang": x["lang"], "kind": x["kind"]})
    with open(f"{out}/{name}.jsonl", "w", encoding="utf-8") as f:
        for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(name, len(rows))
with open(f"{out}/test-B.jsonl", "w", encoding="utf-8") as f:
    n = 0
    for l in open(testb, encoding="utf-8"):
        x = json.loads(l)
        f.write(json.dumps({"src": f"{x['lang']}-itn: {x['spoken']}", "tgt": x["text"], "lang": x["lang"], "kind": x["kind"]},
                           ensure_ascii=False) + "\n"); n += 1
    print("test-B", n)
