"""Run a TN model: python predict_tn.py MODEL TEST.jsonl OUT.jsonl   (envs/pii). Rows have "src" (test-prog) or "lang" + "text" (tn-gap A/B).
Long texts are not split; max_new_tokens = 4x input bytes + 64 (spoken forms are longer than digits)."""
import json, sys, torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1]); model = AutoModelForSeq2SeqLM.from_pretrained(sys.argv[1], dtype=torch.bfloat16).cuda().eval()
test = [json.loads(l) for l in open(sys.argv[2], encoding="utf-8")]
src = [x.get("src") or f'{x["lang"]}-tn: {x["text"]}' for x in test]
order = sorted(range(len(src)), key=lambda i: len(src[i].encode())); pred = [None] * len(src)
with torch.no_grad():
    for i in range(0, len(order), 48):
        ids = order[i:i + 48]
        enc = tok([src[j] for j in ids], return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
        g = model.generate(**enc, max_new_tokens=min(2048, 4 * enc.input_ids.shape[1] + 64), do_sample=False)
        for j, s in zip(ids, tok.batch_decode(g, skip_special_tokens=True)): pred[j] = s
with open(sys.argv[3], "w", encoding="utf-8") as f:
    for p in pred: f.write(json.dumps({"pred": p}, ensure_ascii=False) + "\n")
