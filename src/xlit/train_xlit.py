"""Fine-tune a (By)T5 seq2seq model for romanized -> native transliteration.
python train_xlit.py --model DIR --train train.jsonl --dev dev.jsonl --out OUT [--epochs 1 --bs 32 --lr 3e-4]"""
import argparse, json, math, os, random, time, torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
ap = argparse.ArgumentParser()
for a, d in (("--model", None), ("--train", None), ("--dev", None), ("--out", None), ("--epochs", 1.0), ("--bs", 32), ("--lr", 3e-4),
             ("--max_src", 384), ("--max_tgt", 768), ("--steps", 0)):
    ap.add_argument(a, default=d, type=type(d) if d is not None else str)
A = ap.parse_args()
tok = AutoTokenizer.from_pretrained(A.model); model = AutoModelForSeq2SeqLM.from_pretrained(A.model).cuda()
train = [json.loads(l) for l in open(A.train, encoding="utf-8")]; dev = [json.loads(l) for l in open(A.dev, encoding="utf-8")]
train.sort(key=lambda x: len(x["tgt"].encode()))  # length-bucketed batches, shuffled order
batches = [train[i:i + A.bs] for i in range(0, len(train), A.bs)]
steps = A.steps or math.ceil(len(batches) * A.epochs)
opt = torch.optim.AdamW(model.parameters(), lr=A.lr, weight_decay=0.01)
sch = get_linear_schedule_with_warmup(opt, int(0.03 * steps), steps)
r = random.Random(0); order = []; t0 = time.time(); model.train()
print("train", len(train), "dev", len(dev), "steps", steps, flush=True)
for step in range(steps):
    if not order: order = list(range(len(batches))); r.shuffle(order)
    b = batches[order.pop()]
    enc = tok([x["src"] for x in b], max_length=A.max_src, truncation=True, padding=True, return_tensors="pt").to("cuda")
    lab = tok([x["tgt"] for x in b], max_length=A.max_tgt, truncation=True, padding=True, return_tensors="pt").input_ids.to("cuda")
    lab[lab == tok.pad_token_id] = -100
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**enc, labels=lab).loss
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad()
    if step % 200 == 0: print(f"step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
model.save_pretrained(A.out); tok.save_pretrained(A.out)
json.dump({"steps": steps, "args": {k: v for k, v in vars(A).items() if k not in ("train", "dev", "model", "out")}}, open(A.out + "/train_meta.json", "w"))
print("saved", A.out)
