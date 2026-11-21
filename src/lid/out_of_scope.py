"""Out-of-scope probe: Bhasha-Abhijnaanam manually romanized sentences in 9 languages indic-lid does NOT cover
(as, or, ks, sa, kok, mni, mai, brx, ne). Reports the share labelled as one of our 12 Indic languages.
envs/pii: python out_of_scope.py MODEL_DIR BA.json"""
import json, sys, collections, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1]); model = AutoModelForSequenceClassification.from_pretrained(sys.argv[1]).cuda().eval()
d = [x for x in json.load(open(sys.argv[2], encoding="utf-8"))["data"] if x["source"] == "Manually-Collected" and x["romanized sentence"]]
by = collections.defaultdict(list)
for x in d: by[x["language"]].append(x["romanized sentence"])
for lang, texts in sorted(by.items()):
    for k in (5, 999):
        tt = [" ".join(t.split()[:k]) for t in texts]; pred = []
        with torch.no_grad():
            for i in range(0, len(tt), 256):
                b = tok(tt[i:i + 256], truncation=True, max_length=64, padding=True, return_tensors="pt").to("cuda")
                pred += [model.config.id2label[j] for j in model(**b).logits.argmax(-1).tolist()]
        c = collections.Counter(pred); indic = sum(v for p, v in c.items() if p not in ("en", "other")) / len(pred)
        print(f"{lang:10s} {'5w' if k == 5 else 'full'} n={len(pred)} labelled-Indic {indic:.2f} top {c.most_common(3)}", flush=True)
