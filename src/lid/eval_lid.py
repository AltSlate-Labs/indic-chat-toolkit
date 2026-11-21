"""Score LIDs on the gap-check set. python eval_lid.py TEST.jsonl
fastText models map ISO 639-3 + script labels; xlm-r langdetect uses ISO 639-1 and has no Indic beyond hi/ur."""
import json, os, sys, collections
import fasttext

ISO3 = {"hin": "hi", "ben": "bn", "tam": "ta", "tel": "te", "mar": "mr", "guj": "gu", "kan": "kn", "mal": "ml",
        "pan": "pa", "urd": "ur", "eng": "en", "snd": "sd", "sin": "si"}
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
M = os.path.expanduser("~/Downloads/model/")
models = {"fasttext-lid218": M + "fasttext-language-identification/model.bin", "glotlid": M + "glotlid/model.bin"}
for name, path in models.items():
    m = fasttext.load_model(path); c = collections.Counter(); n = collections.Counter(); wrong = collections.defaultdict(collections.Counter)
    for x in rows:
        lab = m.predict(x["text"].replace("\n", " "))[0][0].replace("__label__", "")
        code = ISO3.get(lab.split("_")[0], lab); key = (x["lang"], x["roman"])
        n[key] += 1; c[key] += code == x["lang"]
        if code != x["lang"]: wrong[key][lab] += 1
    print("==", name)
    for key in sorted(n):
        print(f"  {key[0]:3s} {'roman ' if key[1] else 'native'} {c[key]/n[key]:.2f}  (n={n[key]})  wrong: {dict(wrong[key].most_common(3))}")
    nat = [c[k] / n[k] for k in n if not k[1] and k[0] != "en"]; rom = [c[k] / n[k] for k in n if k[1]]
    print(f"  MEAN native {sum(nat)/len(nat):.3f}  roman {sum(rom)/len(rom):.3f}", flush=True)
