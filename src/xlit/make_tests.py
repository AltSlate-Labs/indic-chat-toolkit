"""Transliteration tests (romanized -> native), built only from held-out LID test data:
  A: Dakshina TEST split sentence pairs (human romanizations), 100 per language, seed 5.
  B: chat pairs from lid-testb (Qwen3.8-27B wrote native first, then romanized it; GlotLID-checked), all 720.
python make_tests.py OUT_DIR"""
import json, os, random, sys, collections
R = os.path.expanduser("~/agent/rnd/jobs/")
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
by = collections.defaultdict(dict)
for l in open(R + "lid-gap/out/dakshina-test.jsonl", encoding="utf-8"):
    x = json.loads(l); by[(x["lang"], x["id"])]["roman" if x["roman"] else "native"] = x["text"]
per = collections.defaultdict(list)
for (lang, _), p in sorted(by.items()):
    if len(p) == 2: per[lang].append({"lang": lang, **p})
r = random.Random(5)
with open(out + "/testA.jsonl", "w", encoding="utf-8") as f:
    for lang in sorted(per):
        for x in r.sample(per[lang], 100): f.write(json.dumps(x, ensure_ascii=False) + "\n")
rows = [json.loads(l) for l in open(R + "lid-testb/out/testb.jsonl", encoding="utf-8")]
with open(out + "/testB.jsonl", "w", encoding="utf-8") as f:
    n = 0
    for a, b in zip(rows[::2], rows[1::2]):
        assert a["lang"] == b["lang"] and not a["roman"] and b["roman"]
        f.write(json.dumps({"lang": a["lang"], "roman": b["text"], "native": a["text"]}, ensure_ascii=False) + "\n"); n += 1
print("A", sum(len(v[:100]) for v in per.values()), "B", n)
