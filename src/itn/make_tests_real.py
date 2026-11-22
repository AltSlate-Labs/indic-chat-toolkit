"""Held-out ITN tests from real sentences (run in envs/tn).
python make_tests_real.py TESTA.jsonl TESTB.jsonl OUT_DIR
 test-A-itn: tn-gap test A (Dakshina VALID split, never trained on) sentences whose numbers are all plain integers; input = spoken
   form made by the same rules as the training wiki rows (correct by construction, our canonical reading style); ref = original.
 testB-teacher.jsonl: test B's teacher-written spoken forms as "pred" rows, to be judged (judge.py) -> make test-B-clean from the
   rows the judge accepts (the teacher's spoken form often reads a different number, e.g. ur 1985 -> "ننانوے سو پچپن")."""
import json, re, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tn"))
from fill_data import Lang, WIKI_BAD, W
a, b, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True); langs = {}
n = 0
with open(f"{out}/test-A-itn.jsonl", "w", encoding="utf-8") as f:
    for l in open(a, encoding="utf-8"):
        x = json.loads(l); t = " ".join(x["text"].split())
        if WIKI_BAD.search(t): continue
        L = langs.setdefault(x["lang"], Lang(x["lang"]))
        rep = lambda m: L.year(int(m.group(0))) if len(m.group(0)) == 4 and 1100 <= int(m.group(0)) <= 2099 else W(int(m.group(0)), x["lang"])
        spoken = re.sub(r"\d+", rep, t); ref = re.sub(r"\d", lambda m: str(int(m.group(0))), t)
        f.write(json.dumps({"src": f"{x['lang']}-itn: {spoken}", "tgt": ref, "lang": x["lang"], "kind": "WIKI"}, ensure_ascii=False) + "\n"); n += 1
print("test-A-itn", n)
with open(f"{out}/testB-teacher.jsonl", "w", encoding="utf-8") as f:
    for l in open(b, encoding="utf-8"): f.write(json.dumps({"pred": json.loads(l)["spoken"]}, ensure_ascii=False) + "\n")
