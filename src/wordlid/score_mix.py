"""Score word-level LID on test-mix (exact labels). python score_mix.py TEST.jsonl NAME=PRED.jsonl ...
word acc (all words, per kind, per gold language); message summary: the set of languages with >= 20% of a row's words
(other excluded) equals the gold set ('mixed detected' for rows whose gold has 2+ languages)."""
import collections, json, sys
test = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
def langs(ls):
    c = collections.Counter(x for x in ls if x != "other"); n = sum(c.values())
    return {k for k, v in c.items() if n and v / n >= 0.2}
for arg in sys.argv[2:]:
    name, f = arg.split("=", 1); P = [json.loads(l)["pred"] for l in open(f, encoding="utf-8")]
    ok = n = 0; kind = collections.defaultdict(lambda: [0, 0]); per = collections.defaultdict(lambda: [0, 0]); sm = [0, 0]; mx = [0, 0]
    for x, p in zip(test, P):
        for g, q in zip(x["labels"], p):
            n += 1; ok += g == q; kind[x["kind"]][0] += g == q; kind[x["kind"]][1] += 1; per[g][0] += g == q; per[g][1] += 1
        G, Q = langs(x["labels"]), langs(p); sm[0] += G == Q; sm[1] += 1
        if len(G) >= 2: mx[0] += G == Q; mx[1] += 1
    print(f"{name:10s} word acc {ok/n:.3f} (n={n})  message langs exact {sm[0]/sm[1]:.3f}  mixed rows exact {mx[0]/mx[1]:.3f} (n={mx[1]})")
    print("  by kind: " + " ".join(f"{k}:{a/b:.3f}" for k, (a, b) in sorted(kind.items())))
    print("  by lang: " + " ".join(f"{k}:{a/b:.2f}" for k, (a, b) in sorted(per.items())))
