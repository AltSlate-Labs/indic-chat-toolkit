"""indic-punct (EkStep, 2022) rule-based ITN baseline. Runs inside the conda container (see notes/itn-plan.md for the recipe).
python baseline_indicpunct.py TEST.jsonl OUT.jsonl   (Urdu is not supported by indic-punct -> input copied)"""
import json, sys, importlib
test = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
by = {}
for i, x in enumerate(test): by.setdefault(x["lang"], []).append(i)
pred = [None] * len(test)
for lang, idx in by.items():
    texts = [test[i]["src"].split(": ", 1)[1] for i in idx]
    try: out = importlib.import_module(f"inverse_text_normalization.{lang}.run_predict").inverse_normalize_text(texts)
    except ModuleNotFoundError: out = texts
    for i, o in zip(idx, out): pred[i] = " ".join(o.split())
with open(sys.argv[2], "w", encoding="utf-8") as f:
    for p in pred: f.write(json.dumps({"pred": p}, ensure_ascii=False) + "\n")
