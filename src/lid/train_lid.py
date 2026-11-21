"""python train_lid.py --model DIR --train a.jsonl --dev b.jsonl --out DIR [--epochs 1]"""
import argparse, json, math, random, time, collections
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from synth_lid import LABELS

L2I = {l: i for i, l in enumerate(LABELS)}


def load(p): return [json.loads(x) for x in open(p, encoding="utf-8")]


def batches(tok, rows, bs, max_len, shuffle):
    idx = list(range(len(rows)))
    if shuffle: random.shuffle(idx)
    for i in range(0, len(idx), bs):
        b = [rows[j] for j in idx[i:i + bs]]
        enc = tok([x["text"] for x in b], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([L2I[x["label"]] for x in b])
        yield enc


def evaluate(model, tok, rows):
    model.eval(); ok = collections.Counter(); n = collections.Counter()
    with torch.no_grad():
        for b in batches(tok, rows, 256, 64, False):
            y = b.pop("labels"); b = {k: v.cuda() for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model(**b).logits.argmax(-1).cpu()
            for a, g in zip(p.tolist(), y.tolist()):
                n[LABELS[g]] += 1; ok[LABELS[g]] += a == g
    model.train()
    return sum(ok.values()) / sum(n.values()), {k: round(ok[k] / n[k], 3) for k in sorted(n)}


def main():
    ap = argparse.ArgumentParser()
    for a in ("--model", "--train", "--dev", "--out"): ap.add_argument(a)
    ap.add_argument("--epochs", type=float, default=1); ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=128); ap.add_argument("--max_len", type=int, default=64)
    a = ap.parse_args(); torch.manual_seed(0); random.seed(0)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=len(LABELS),
            id2label=dict(enumerate(LABELS)), label2id=L2I).cuda()
    tr, dv = load(a.train), load(a.dev)
    steps = math.ceil(len(tr) / a.bs * a.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    step, t0 = 0, time.time(); print(f"train {len(tr)} dev {len(dv)} steps {steps}", flush=True)
    while step < steps:
        for b in batches(tok, tr, a.bs, a.max_len, True):
            b = {k: v.cuda() for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**b).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(); step += 1
            if step % 200 == 0: print(f"step {step}/{steps} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
            if step >= steps: break
    acc, per = evaluate(model, tok, dv)
    print(f"DEV acc {acc:.4f}\n{per}", flush=True)
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    json.dump({"dev_acc": acc, "dev_per": per, "steps": steps, "args": vars(a)}, open(f"{a.out}/train_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
