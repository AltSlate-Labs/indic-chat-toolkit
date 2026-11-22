"""Punctuation restoration: fine-tune an encoder for token classification on {"words", "labels"} rows (label on each word's first sub-token).
python train_wordlid.py --model DIR --train train.jsonl --dev dev.jsonl --out OUT [--epochs 1 --bs 32 --lr 5e-5]
Checkpoints every --ckpt_every steps to OUT/ckpt.pt; rerun with the same --out to resume."""
import argparse, json, math, os, random, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_MEM_FRAC", "0.4")))
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup
from synth_punct import LABELS

ap = argparse.ArgumentParser()
for a, d in (("--model", None), ("--train", None), ("--dev", None), ("--out", None), ("--epochs", 1.0), ("--bs", 32), ("--lr", 5e-5),
             ("--max_len", 256), ("--ckpt_every", 1000)):
    ap.add_argument(a, default=d, type=type(d) if d is not None else str)
A = ap.parse_args()
L2I = {l: i for i, l in enumerate(LABELS)}
tok = AutoTokenizer.from_pretrained(A.model)
model = AutoModelForTokenClassification.from_pretrained(A.model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=L2I).cuda()


def encode(rows):
    enc = tok([x["words"] for x in rows], is_split_into_words=True, truncation=True, max_length=A.max_len, padding=True, return_tensors="pt")
    lab = torch.full(enc.input_ids.shape, -100)
    for b, x in enumerate(rows):
        prev = None
        for p, w in enumerate(enc.word_ids(b)):
            if w is not None and w != prev: lab[b, p] = L2I[x["labels"][w]]
            prev = w
    return enc, lab


def evaluate(rows):
    model.eval(); ok = n = 0
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            enc, lab = encode(rows[i:i + 64]); pred = model(**enc.to("cuda")).logits.argmax(-1).cpu()
            m = lab != -100; ok += int((pred[m] == lab[m]).sum()); n += int(m.sum())
    model.train(); return ok / n


train = [json.loads(l) for l in open(A.train, encoding="utf-8")]; dev = [json.loads(l) for l in open(A.dev, encoding="utf-8")]
train.sort(key=lambda x: len(x["words"]))
batches = [train[i:i + A.bs] for i in range(0, len(train), A.bs)]; steps = math.ceil(len(batches) * A.epochs)
opt = torch.optim.AdamW(model.parameters(), lr=A.lr, weight_decay=0.01); sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
r = random.Random(0); order = []; start = 0; ck = A.out + "/ckpt.pt"; os.makedirs(A.out, exist_ok=True)
if os.path.exists(ck):
    c = torch.load(ck, weights_only=False); model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sch.load_state_dict(c["sch"])
    start, order = c["step"], c["order"]; r.setstate(c["rng"]); print("resumed at step", start, flush=True)
print("train", len(train), "dev", len(dev), "steps", steps, flush=True); t0 = time.time(); model.train()
for step in range(start, steps):
    if step and step % A.ckpt_every == 0:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(), "step": step, "order": order,
                    "rng": r.getstate()}, ck + ".tmp"); os.replace(ck + ".tmp", ck)
    if not order: order = list(range(len(batches))); r.shuffle(order)
    enc, lab = encode(batches[order.pop()])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**enc.to("cuda"), labels=lab.cuda()).loss
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad()
    if step % 200 == 0: print(f"step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
print("dev word accuracy", round(evaluate(dev), 4), flush=True)
model.save_pretrained(A.out); tok.save_pretrained(A.out)
if os.path.exists(ck): os.remove(ck)
print("saved", A.out)
