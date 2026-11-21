"""CER (char edit distance / reference length, NFC, whitespace-normalized) and word accuracy, per language.
python metrics.py TEST.jsonl PRED.jsonl   (PRED rows: {"pred": ...} in test order)"""
import json, sys, unicodedata, collections


def norm(s): return " ".join(unicodedata.normalize("NFC", s).split())


def ed(a, b):
    p = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        c = [i]
        for j, cb in enumerate(b, 1): c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (ca != cb)))
        p = c
    return p[-1]


def score(test, preds):
    e, n, wa, wn = (collections.Counter() for _ in range(4))
    for x, p in zip(test, preds):
        ref, hyp = norm(x["native"]), norm(p); L = x["lang"]
        e[L] += ed(ref, hyp); n[L] += len(ref)
        rw, hw = ref.split(), hyp.split(); wn[L] += len(rw); wa[L] += sum(a == b for a, b in zip(rw, hw)) if len(rw) == len(hw) else 0
    cer = {L: e[L] / n[L] for L in n}
    return sum(cer.values()) / len(cer), cer, {L: wa[L] / wn[L] for L in wn}


if __name__ == "__main__":
    test = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
    preds = [json.loads(l)["pred"] for l in open(sys.argv[2], encoding="utf-8")]
    m, cer, wacc = score(test, preds)
    print(f"CER {m:.3f} | " + " ".join(f"{L}:{v:.3f}" for L, v in sorted(cer.items())) + " | word-acc " + " ".join(f"{L}:{v:.2f}" for L, v in sorted(wacc.items())))
