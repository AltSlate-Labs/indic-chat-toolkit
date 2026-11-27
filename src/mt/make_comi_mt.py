"""COMI-LINGUA MT (LingoIITGN, CC-BY-4.0; news-style Hindi-English code-mixed, Llama-3.3 drafts corrected by 3 annotators) ->
{src, ref, refs, kind} rows. kind "roman" = annotator 1 romanized-Hindi version, "mixed" = original Devanagari+English sentence.
Refs = the 3 annotator English translations (deduped). Used for EVALUATION ONLY (not trained on).
python make_comi_mt.py train|test OUT_JSONL N  (dev = first N of a seed-11 shuffle of MT_train; test = MT_test)"""
import json, os, random, sys, pandas as pd
split, out, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = pd.read_csv(os.path.expanduser(f"~/Downloads/data/comi-lingua/MT_{split}.csv")).astype(str)
ix = list(range(len(d))); random.Random(11).shuffle(ix)
with open(out, "w", encoding="utf-8") as f:
    for i in ix[:N]:
        r = d.iloc[i]
        refs = list(dict.fromkeys(r[f"Annotator_{k}_en_translation"].strip() for k in (1, 2, 3)))
        for kind, src in (("roman", r["Annotator_1_RH_translation"]), ("mixed", r["Sentences"])):
            f.write(json.dumps({"src": src.strip(), "ref": refs[0], "refs": refs, "kind": kind, "lang": "hi"}, ensure_ascii=False) + "\n")
