"""Compare indic-lid with GlotLID on test A (Dakshina test, cut to k words) and test B (chat).
Two steps, because fasttext (envs/lid, numpy<2) and torch (envs/pii) live in different venvs:
  envs/lid:  python compare.py glot A.jsonl B.jsonl CACHE.json
  envs/lid:  python compare.py indiclid A.jsonl B.jsonl ICACHE.json   (IndicLID FTN/FTR, no BERT stage)
  envs/pii:  python compare.py MODEL_DIR A.jsonl B.jsonl CACHE.json [ICACHE.json]"""
import json, os, sys, collections

ISO3 = {"hin": "hi", "ben": "bn", "tam": "ta", "tel": "te", "mar": "mr", "guj": "gu", "kan": "kn", "mal": "ml",
        "pan": "pa", "urd": "ur", "snd": "sd", "sin": "si"}
GLOT = sys.argv[1] in ("glot", "indiclid"); IND = sys.argv[1] == "indiclid"
if GLOT:
    import fasttext
    M = os.path.expanduser("~/Downloads/model/IndicLID/IndicLID-%s/model_baseline_roman.bin")
    if IND: ftr, ftn = fasttext.load_model(M % "FTR"), fasttext.load_model(M % "FTN")
    else: glot = fasttext.load_model(os.path.expanduser("~/Downloads/model/glotlid/model.bin"))
    CACHE = {}
else:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(sys.argv[1]); model = AutoModelForSequenceClassification.from_pretrained(sys.argv[1]).cuda().eval()
    CACHE = json.load(open(sys.argv[4]))
    ICACHE = json.load(open(sys.argv[5])) if len(sys.argv) > 5 else None


def i_pred(texts, name):
    return ICACHE[name]


def g_pred(texts, name):
    if not GLOT: return CACHE[name]
    out = []
    for t in texts:
        t = " ".join(t.split())
        if IND:  # IndicLID routing: mostly Latin letters -> roman model, else native model
            lat = sum(c.isascii() and c.isalpha() for c in t); al = sum(c.isalpha() for c in t)
            labs = (ftr if lat > al / 2 else ftn).predict(t, k=-1)[0]
            if os.environ.get("IND_RESTRICT"):  # generous: best label among our 12 languages + en
                labs = [x for x in labs if x.split("_")[2] in ISO3 or x.startswith("__label__eng")] or labs
            lab = labs[0].replace("__label__", "")
        else: lab = glot.predict(t)[0][0].replace("__label__", "")
        iso, scr = (lab.split("_") + [""])[:2]; l = ISO3.get(iso)
        out.append(l + ("-Latn" if scr == "Latn" else "") if l else ("en" if iso == "eng" else "other"))
    CACHE[name] = out
    return out


def m_pred(texts, name):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 256):
            b = tok(texts[i:i + 256], truncation=True, max_length=64, padding=True, return_tensors="pt").to("cuda")
            out += [model.config.id2label[j] for j in model(**b).logits.argmax(-1).tolist()]
    return out


def report(name, rows):
    gold = [x["lang"] + ("-Latn" if x["roman"] else "") for x in rows]
    systems = (("glotlid", g_pred),) if GLOT else (("glotlid", g_pred),) + ((("indiclid", i_pred),) if ICACHE else ()) + (("indic-lid", m_pred),)
    for sys_name, f in systems:
        pred = f([x["text"] for x in rows], name); c = collections.Counter(); n = collections.Counter(); cm = collections.Counter()
        for p, g in zip(pred, gold):
            n[g] += 1; c[g] += p == g
            cm[g] += p == g or {p, g} == {"hi-Latn", "ur-Latn"}
        nat = [c[k] / n[k] for k in n if not k.endswith("-Latn")]; rom = [c[k] / n[k] for k in n if k.endswith("-Latn")]
        romm = [cm[k] / n[k] for k in n if k.endswith("-Latn")]
        per = " ".join(f"{k[:2]}:{c[k]/n[k]:.2f}" for k in sorted(n) if k.endswith("-Latn"))
        print(f"{name:14s} {sys_name:9s} native {sum(nat)/len(nat):.3f}  roman {sum(rom)/len(rom):.3f}  "
              f"roman(hi=ur) {sum(romm)/len(romm):.3f} | roman per lang {per}", flush=True)


A = [json.loads(l) for l in open(sys.argv[2], encoding="utf-8")]
for k in (3, 5, 8, 999):
    report(f"A first {k}w" if k < 999 else "A full", [{**x, "text": " ".join(x["text"].split()[:k])} for x in A])
report("B chat", [json.loads(l) for l in open(sys.argv[3], encoding="utf-8")])
if GLOT: json.dump(CACHE, open(sys.argv[4], "w"))
