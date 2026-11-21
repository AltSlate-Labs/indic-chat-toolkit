"""Test C: messages from the PII project's dev splits (written by the test teacher, never used for LID training; they contain
fake IDs, phones and e-mails like real support chat). Gold: en -> en, hinglish -> hi-Latn, hi/mr/bn/ta/te/gu/kn -> native.
envs/lid: python test_c.py glot      envs/pii: python test_c.py MODEL_DIR"""
import json, os, sys, collections
R = os.path.expanduser("~/agent/rnd/jobs/")
rows = []
for f in ("testgen-v0/out/dev.jsonl", "testgen-v5/out/dev.jsonl"):
    for l in open(R + f, encoding="utf-8"):
        x = json.loads(l); lang = x["lang"].replace("-neg", "")
        rows.append((x["text"], {"en": "en", "hinglish": "hi-Latn"}.get(lang, lang)))
texts = [t for t, _ in rows]
if sys.argv[1] == "glot":
    import fasttext
    ISO3 = {"hin": "hi", "ben": "bn", "tam": "ta", "tel": "te", "mar": "mr", "guj": "gu", "kan": "kn", "mal": "ml",
            "pan": "pa", "urd": "ur", "snd": "sd", "sin": "si"}
    m = fasttext.load_model(os.path.expanduser("~/Downloads/model/glotlid/model.bin")); pred = []
    for t in texts:
        lab = m.predict(" ".join(t.split()))[0][0][9:]; iso, scr = lab.split("_"); l = ISO3.get(iso)
        pred.append(l + ("-Latn" if scr == "Latn" else "") if l else ("en" if iso == "eng" else "other"))
    name = "glotlid"
else:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(sys.argv[1]); model = AutoModelForSequenceClassification.from_pretrained(sys.argv[1]).cuda().eval(); pred = []
    with torch.no_grad():
        for i in range(0, len(texts), 128):
            b = tok(texts[i:i + 128], truncation=True, max_length=64, padding=True, return_tensors="pt").to("cuda")
            pred += [model.config.id2label[j] for j in model(**b).logits.argmax(-1).tolist()]
    name = os.path.basename(os.path.dirname(os.path.dirname(sys.argv[1].rstrip("/"))))
n, c, wrong = collections.Counter(), collections.Counter(), collections.defaultdict(collections.Counter)
for p, (_, g) in zip(pred, rows):
    n[g] += 1; c[g] += p == g
    if p != g: wrong[g][p] += 1
print(f"C {name:8s} " + " ".join(f"{g}:{c[g]/n[g]:.3f}(n={n[g]})" for g in sorted(n)), "| errors", {g: dict(w.most_common(3)) for g, w in wrong.items()})
