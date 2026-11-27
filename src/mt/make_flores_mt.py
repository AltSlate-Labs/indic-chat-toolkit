"""FLORES (IndicGenBench bitext) Indic -> English splits for MT: native-script source and the same source romanized by our
dictionary+rule romanizer (seed 7, as in src/embed/evaluate.py; synthetic, favours our model). N sentences per language.
python make_flores_mt.py SPLIT(dev|test) OUT_JSONL N"""
import json, os, sys, pyarrow.parquet as pq
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "embed"))
from romanizer import Romanizer
split, out, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
D = os.path.expanduser("~/Downloads/data"); R = Romanizer(seed=7)
ISO3 = dict(hi="hin", bn="ben", ta="tam", te="tel", mr="mar", gu="guj", kn="kan", ml="mal", pa="pan", ur="urd")
with open(out, "w", encoding="utf-8") as f:
    for l, c in ISO3.items():
        t = pq.read_table(f"{D}/IndicGenBenchFloresBitextMining/{c}-eng/{'test' if split == 'test' else 'validation'}-00000-of-00001.parquet").to_pylist()[:N]
        for r in t:
            f.write(json.dumps({"src": r["sentence1"], "ref": r["sentence2"], "lang": l, "kind": "native"}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"src": R.sentence(r["sentence1"], l), "ref": r["sentence2"], "lang": l, "kind": "roman"}, ensure_ascii=False) + "\n")
print("wrote", out)
