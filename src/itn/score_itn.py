"""Score ITN outputs. python score_itn.py TEST.jsonl NAME=PRED.jsonl ...  (PRED rows: {"pred": ...})
 exact : output == reference (whitespace-normalized); meaningful only for programmatic tests with canonical references.
 digits: the output's digits, in order (native digits mapped to ASCII, all separators dropped), equal the reference's:
         format-agnostic, so ₹1,250 / Rs 1250 and 98765 43210 / 9876543210 count as equal; also wrong if digits are missing.
 Reported overall, per language, and without ordinals (references like "3वें" have digits, the canonical ITN keeps ordinal words)."""
import json, sys, collections
test = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
D = lambda s: "".join(str(int(c)) for c in s if c.isdecimal())
N = lambda s: " ".join(s.split())
for arg in sys.argv[2:]:
    name, f = arg.split("=", 1); pred = [json.loads(l)["pred"] for l in open(f, encoding="utf-8")]
    ex = [N(p) == N(x["tgt"]) for p, x in zip(pred, test)]; dg = [D(p) == D(x["tgt"]) for p, x in zip(pred, test)]
    keep = [i for i, x in enumerate(test) if "ordinal" not in x["kind"].lower()]
    by = collections.defaultdict(list)
    for x, d in zip(test, dg): by[x["lang"]].append(d)
    print(f"{name:10s} exact {sum(ex)/len(ex):.3f}  digits {sum(dg)/len(dg):.3f}  digits-no-ordinal {sum(dg[i] for i in keep)/len(keep):.3f} (n={len(test)}) | "
          + " ".join(f"{l}:{sum(v)/len(v):.2f}" for l, v in sorted(by.items())))
