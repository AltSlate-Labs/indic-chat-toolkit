"""GlotLID accuracy on romanized Dakshina test sentences cut to the first k words. python by_length.py TEST.jsonl"""
import json, os, sys, collections
import fasttext
ISO3 = {"hin": "hi", "ben": "bn", "tam": "ta", "tel": "te", "mar": "mr", "guj": "gu", "kan": "kn", "mal": "ml",
        "pan": "pa", "urd": "ur", "snd": "sd", "sin": "si"}
m = fasttext.load_model(os.path.expanduser("~/Downloads/model/glotlid/model.bin"))
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
rows = [x for x in rows if x["roman"] and x["lang"] not in ("si",)]
for k in (3, 5, 8, 12, 999):
    c = collections.Counter(); n = collections.Counter()
    for x in rows:
        t = " ".join(x["text"].split()[:k]); lab = m.predict(t)[0][0].replace("__label__", "")
        code = ISO3.get(lab.split("_")[0], lab); n[x["lang"]] += 1
        c[x["lang"]] += code == x["lang"] or {code, x["lang"]} == {"hi", "ur"}
    print(f"first {k:>3} words: mean {sum(c.values())/sum(n.values()):.3f} | " + " ".join(f"{l} {c[l]/n[l]:.2f}" for l in sorted(n)), flush=True)
