"""Pick no_repeat_ngram_size (byte n-grams) on the DEV split only: python sweep_nrng.py MODEL DEV.jsonl
Counts looping outputs (pred > 1.3x ref + 5 chars) and macro CER for default-mode dev rows with >= 40 input chars."""
import json, sys, torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from metrics import score
tok = AutoTokenizer.from_pretrained(sys.argv[1]); model = AutoModelForSeq2SeqLM.from_pretrained(sys.argv[1], dtype=torch.bfloat16).cuda().eval()
dev = []
for l in open(sys.argv[2], encoding="utf-8"):
    x = json.loads(l); pre, rom = x["src"].split(": ", 1)
    if "+" not in pre and len(rom) >= 40: dev.append({"lang": x["lang"], "roman": rom, "native": x["tgt"]})
print("dev rows", len(dev), flush=True)
order = sorted(range(len(dev)), key=lambda i: len(dev[i]["roman"]))
for n in (0, 16, 24, 32, 48):
    pred = [None] * len(dev)
    with torch.no_grad():
        for i in range(0, len(order), 64):
            ids = order[i:i + 64]
            enc = tok([f'{dev[j]["lang"]}: {dev[j]["roman"]}' for j in ids], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            kw = {"no_repeat_ngram_size": n} if n else {}
            g = model.generate(**enc, max_new_tokens=min(1024, 4 * enc.input_ids.shape[1] + 32), do_sample=False, **kw)
            for j, s in zip(ids, tok.batch_decode(g, skip_special_tokens=True)): pred[j] = s
    loops = sum(len(p) > 1.3 * len(x["native"]) + 5 for x, p in zip(dev, pred))
    print(f"nrng {n:2d} CER {score(dev, pred)[0]:.4f} loops {loops}", flush=True)
