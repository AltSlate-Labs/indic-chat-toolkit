"""Distil bge-m3 into mmBERT-small for native / romanized / code-mixed Indic + English sentences (notes/embed-plan.md).
python train_embed.py --data DIR --base DIR --out DIR [--epochs 1] [--bs 128] [--lr 5e-5] [--max_steps N] [--ckpt_every 2000]
Student: mean pooling over tokens -> Linear(384 -> 1024) -> L2 norm. Loss: MSE to the teacher vector of the row's target
+ in-batch InfoNCE (student vs teacher targets, temperature 0.05; other rows with the same target are masked, not negatives).
Resume: rerun with the same --out (OUT/ckpt.pt). Output: OUT/ (HF model + tokenizer) and OUT/head.pt."""
import argparse, json, os, random, time, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
p = argparse.ArgumentParser()
for a, d in [("--data", None), ("--base", None), ("--out", None), ("--epochs", 1.0), ("--bs", 128), ("--lr", 5e-5), ("--max_steps", 0),
             ("--ckpt_every", 2000), ("--max_len", 128), ("--mse_w", 1.0), ("--nce_w", 1.0), ("--init_head", ""), ("--chunk_steps", 0)]:
    p.add_argument(a, type=type(d) if d is not None else str, default=d)
A = p.parse_args(); os.makedirs(A.out, exist_ok=True)
torch.manual_seed(0); random.seed(0)
texts = json.load(open(f"{A.data}/targets.json", encoding="utf-8")); T = np.load(f"{A.data}/targets.f16.npy", mmap_mode="r")
idx = {t: i for i, t in enumerate(texts)}
def load(f):
    rows = []
    for l in open(f"{A.data}/{f}.jsonl", encoding="utf-8"):
        x = json.loads(l); s = x["input"]
        if x["kind"] == "roman": s = s.replace("।", ".").replace("॥", ".")   # romanized chat uses "." not the danda
        rows.append((s, idx[x["target"]]))
    return rows
train, dev = load("train"), load("dev")
tok = AutoTokenizer.from_pretrained(A.base)
enc = AutoModel.from_pretrained(A.base, torch_dtype=torch.float32).cuda()
head = torch.nn.Linear(enc.config.hidden_size, 1024).cuda()
if A.init_head: head.load_state_dict(torch.load(A.init_head, map_location="cuda"))   # fine-tune from an earlier student
params = [{"params": enc.parameters(), "lr": A.lr}, {"params": head.parameters(), "lr": A.lr * 20}]
opt = torch.optim.AdamW(params, weight_decay=0.01)
steps = A.max_steps or int(len(train) * A.epochs / A.bs)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * max(0.0, 1 - s / steps))
def embed(batch):
    b = tok(batch, padding=True, truncation=True, max_length=A.max_len, return_tensors="pt").to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = enc(**b).last_hidden_state
    mask = b["attention_mask"][..., None].float()
    return torch.nn.functional.normalize(head((h.float() * mask).sum(1) / mask.sum(1)), dim=-1)
step = 0; ck = f"{A.out}/ckpt.pt"
if os.path.exists(ck):
    c = torch.load(ck, map_location="cuda"); enc.load_state_dict(c["enc"]); head.load_state_dict(c["head"]); opt.load_state_dict(c["opt"])
    sched.load_state_dict(c["sched"]); step = c["step"]; print("resumed at step", step, flush=True)
order = list(range(len(train))); random.Random(1).shuffle(order)
def dev_eval():
    enc.eval(); head.eval(); cos = []
    with torch.no_grad():
        for i in range(0, 2000, 256):
            rows = dev[i:i + 256]; s = embed([r[0] for r in rows]); t = torch.tensor(np.stack([T[r[1]] for r in rows]), device="cuda").float()
            cos.append((s * t).sum(-1))
    enc.train(); head.train(); return torch.cat(cos).mean().item()
print(f"train {len(train)} dev {len(dev)} steps {steps} bs {A.bs}", flush=True)
t0 = time.time(); enc.train()
stop = min(steps, step + A.chunk_steps) if A.chunk_steps else steps   # chunked runs: stop on a ckpt boundary, rerun to resume
while step < stop:
    i0 = (step * A.bs) % len(train); rows = [train[order[(i0 + k) % len(train)]] for k in range(A.bs)]
    tid = torch.tensor([r[1] for r in rows], device="cuda")
    t = torch.tensor(np.stack([T[r[1]] for r in rows]), device="cuda").float()
    s = embed([r[0] for r in rows])
    mse = ((s - t) ** 2).sum(-1).mean()
    logits = s @ t.T / 0.05
    same = (tid[:, None] == tid[None, :]) & ~torch.eye(len(rows), dtype=torch.bool, device="cuda")
    nce = torch.nn.functional.cross_entropy(logits.masked_fill(same, -1e4), torch.arange(len(rows), device="cuda"))
    loss = A.mse_w * mse + A.nce_w * nce
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0); opt.step(); sched.step(); step += 1
    if step % 100 == 0 or step == 1:
        print(f"step {step}/{steps} loss {loss.item():.4f} mse {mse.item():.4f} nce {nce.item():.3f} {time.time() - t0:.0f}s", flush=True)
    if step % A.ckpt_every == 0 or step == steps:
        print(f"step {step} dev cos {dev_eval():.4f}", flush=True)
        torch.save({"enc": enc.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, ck)
if step < steps: print(f"chunk done at step {step}/{steps}; rerun to resume", flush=True); raise SystemExit(0)
enc.save_pretrained(A.out); tok.save_pretrained(A.out); torch.save(head.state_dict(), f"{A.out}/head.pt")
json.dump({"steps": steps, "bs": A.bs, "lr": A.lr, "data": A.data}, open(f"{A.out}/train_meta.json", "w"))
print("saved", A.out)
