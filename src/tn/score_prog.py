"""Exact-match score on the programmatic test (held-out carriers): python score_prog.py TEST-PROG.jsonl NAME=PRED.jsonl [...]
Exact = prediction equals the gold spoken sentence after whitespace normalization. Also prints per-kind and per-language exact match."""
import json, sys, collections
test = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
N = lambda s: " ".join(s.split())
for arg in sys.argv[2:]:
    name, path = arg.split("=", 1); pred = [json.loads(l)["pred"] for l in open(path, encoding="utf-8")]
    k, kn, l, ln = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    for x, p in zip(test, pred):
        ok = N(p) == N(x["tgt"]); k[x["kind"]] += ok; kn[x["kind"]] += 1; l[x["lang"]] += ok; ln[x["lang"]] += 1
    print(f"{name:10s} exact {sum(k.values()) / len(test):.3f} (n={len(test)})")
    print("  per kind:", " ".join(f"{a}:{k[a] / kn[a]:.2f}" for a in sorted(kn)))
    print("  per lang:", " ".join(f"{a}:{l[a] / ln[a]:.2f}" for a in sorted(ln)))
