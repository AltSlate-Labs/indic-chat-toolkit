"""Transliteration with a trained model: python predict.py MODEL TEST.jsonl OUT.jsonl [BEAMS=1] [PREFIX_SUFFIX=""]"""
import json, sys, torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1]); model = AutoModelForSeq2SeqLM.from_pretrained(sys.argv[1], dtype=torch.bfloat16).cuda().eval()
test = [json.loads(l) for l in open(sys.argv[2], encoding="utf-8")]
beams = int(sys.argv[4]) if len(sys.argv) > 4 else 1; suf = sys.argv[5] if len(sys.argv) > 5 else ""
order = sorted(range(len(test)), key=lambda i: len(test[i]["roman"])); pred = [None] * len(test)
with torch.no_grad():
    for i in range(0, len(order), 64 // beams if beams > 1 else 64):
        ids = order[i:i + (64 // beams if beams > 1 else 64)]
        enc = tok([f'{test[j]["lang"]}{suf}: {test[j]["roman"]}' for j in ids], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        g = model.generate(**enc, max_new_tokens=min(1024, 4 * enc.input_ids.shape[1] + 32), do_sample=False, num_beams=beams)
        for j, s in zip(ids, tok.batch_decode(g, skip_special_tokens=True)): pred[j] = s
with open(sys.argv[3], "w", encoding="utf-8") as f:
    for p in pred: f.write(json.dumps({"pred": p}, ensure_ascii=False) + "\n")
