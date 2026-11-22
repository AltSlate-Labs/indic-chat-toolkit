"""Baseline: PhysicsWallahAI/muril-hinglish-lid (HIN/ENG per word, first sub-token), as in its model card.
python baseline_muril.py MODEL_DIR TEST.jsonl OUT.jsonl"""
import json, sys, torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
d, test, out = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(d); m = AutoModelForTokenClassification.from_pretrained(d).eval().to("cuda" if torch.cuda.is_available() else "cpu")
MAP = {"HIN": "hi", "ENG": "en"}
with open(out, "w", encoding="utf-8") as f:
    for l in open(test, encoding="utf-8"):
        words = json.loads(l)["words"]; labs = []
        for i in range(0, len(words), 120):  # chunk long rows (model max 256 sub-tokens)
            w = words[i:i + 120]; enc = tok(w, is_split_into_words=True, truncation=True, max_length=512, return_tensors="pt").to(m.device)
            with torch.no_grad(): pr = m(**enc).logits[0].argmax(-1).tolist()
            got = {}
            for pos, wid in enumerate(enc.word_ids()):
                if wid is not None and wid not in got: got[wid] = MAP[m.config.id2label[pr[pos]]]
            labs += [got.get(j, "en") for j in range(len(w))]
        f.write(json.dumps({"pred": labs}) + "\n")
