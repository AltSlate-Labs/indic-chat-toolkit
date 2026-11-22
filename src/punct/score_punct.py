"""Punctuation F1. python score_punct.py TEST.jsonl NAME=PRED.jsonl ...   (PRED rows {"pred": [class per word]} or {"skip": true})
Per class precision/recall/F1 over words (micro), macro-F1 over COMMA/PERIOD/QUESTION; per-language macro-F1. Rows a system could not
align (word count changed) count as all-O predictions."""
import collections, json, sys
test = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
C = ["COMMA", "PERIOD", "QUESTION"]
def f1(tp, fp, fn): p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0; return 2 * p * r / (p + r) if p + r else 0
for arg in sys.argv[2:]:
    name, f = arg.split("=", 1); P = [json.loads(l) for l in open(f, encoding="utf-8")]
    cnt = collections.defaultdict(lambda: [0, 0, 0]); per = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0])); skip = 0
    for x, p in zip(test, P):
        pr = p.get("pred") or ["O"] * len(x["labels"]); skip += "pred" not in p
        for g, q in zip(x["labels"], pr):
            for c in C:
                a = [cnt[c], per[x["lang"]][c]]
                for z in a:
                    if q == c and g == c: z[0] += 1
                    elif q == c: z[1] += 1
                    elif g == c: z[2] += 1
    G = [c for c in C if cnt[c][0] + cnt[c][2] > 0]  # macro over classes present in the gold labels (test-chat has no questions)
    F = {c: f1(*cnt[c]) for c in C}
    print(f"{name:14s} macro-F1 {sum(F[c] for c in G)/len(G):.3f} | " + " ".join(f"{c} {F[c]:.3f}" for c in G) + f" | unaligned rows {skip}")
    pl = lambda d: [c for c in C if d[c][0] + d[c][2] > 0]
    print("   per lang macro-F1: " + " ".join(f"{l}:{sum(f1(*d[c]) for c in pl(d))/max(1, len(pl(d))):.2f}" for l, d in sorted(per.items())))
