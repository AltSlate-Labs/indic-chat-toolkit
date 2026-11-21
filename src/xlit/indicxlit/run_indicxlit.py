"""Baseline: AI4Bharat IndicXlit en->indic (MIT), word-level, official preprocessing (lowercase, space-separated chars,
__lang__ tag), beam 4, top-1, WITHOUT the word-probability rescoring (850 MB dicts not fetched). Latin words in a sentence are
transliterated one by one; digits and punctuation are kept. Runs inside python:3.10 docker with fairseq 0.12.2.
python run_indicxlit.py MODEL_DIR TEST.jsonl OUT.jsonl"""
import json, os, re, sys
mdir, test_f, out_f = sys.argv[1:4]
here = os.path.dirname(os.path.abspath(__file__))
os.chdir(mdir)  # corpus-bin/, transformer/indicxlit.pt
import shutil; shutil.copy(os.path.join(here, "lang_list.txt"), "lang_list.txt")
sys.argv = [sys.argv[0]]
from custom_interactive import Transliterator
tr = Transliterator("corpus-bin", "transformer/indicxlit.pt", 4, 1, batch_size=128)
test = [json.loads(l) for l in open(test_f, encoding="utf-8")]
W = re.compile(r"[A-Za-z]+")
need = sorted({(x["lang"], w.lower()) for x in test for w in W.findall(x["roman"])})
print("unique words", len(need), flush=True)
best = {}
for i in range(0, len(need), 2000):
    chunk = need[i:i + 2000]
    res = tr.translate([f"__{l}__ " + " ".join(w) for l, w in chunk])
    hyp = {}
    for line in res.split("\n"):
        if line.startswith("H-"):
            k, _, toks = line.split("\t"); k = int(k[2:])
            if k not in hyp: hyp[k] = "".join(t for t in toks.split() if not t.startswith("__"))
    for k, (l, w) in enumerate(chunk): best[(l, w)] = hyp.get(k, w)
with open(out_f, "w", encoding="utf-8") as f:
    for x in test:
        f.write(json.dumps({"pred": W.sub(lambda m: best[(x["lang"], m.group(0).lower())], x["roman"])}, ensure_ascii=False) + "\n")
