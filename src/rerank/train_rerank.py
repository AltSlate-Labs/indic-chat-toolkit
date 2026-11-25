"""Distil bge-reranker-v2-m3 into an mmBERT cross-encoder for native AND romanized Indic queries (notes/rerank-plan.md).
python train_rerank.py --lists FILE --base DIR --out DIR [--steps 6000] [--q 16] [--g 16] [--lr 3e-5] [--roman 0.5] [--mixed 0.15]
Each step: Q queries x G passages (judged positives + highest-teacher + random rest). Each query is shown native, fully romanized
(prob --roman) or half-romanized (prob --mixed) by src/embed/romanizer.py; the target is always the teacher's scores for the
NATIVE query (the teacher cannot read romanized queries: MIRACLReranking romanized nDCG .560 ~ no reranking).
Loss: listwise KL(softmax(teacher) || softmax(student)) + 0.1 * MSE(student, teacher). 3% of queries held out: reports
listwise KL and top-1 agreement with the teacher (native and romanized). Checkpoint/resume: OUT/ckpt.pt."""
import argparse, json, os, random, sys, time, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "embed"))
from romanizer import Romanizer
p = argparse.ArgumentParser()
for a, d in [("--lists", None), ("--base", None), ("--out", None), ("--steps", 6000), ("--q", 16), ("--g", 16), ("--lr", 3e-5),
             ("--roman", 0.5), ("--mixed", 0.15), ("--max_len", 320), ("--ckpt_every", 1000), ("--chunk_steps", 0)]:
    p.add_argument(a, type=type(d) if d is not None else str, default=d)
A = p.parse_args(); os.makedirs(A.out, exist_ok=True); torch.manual_seed(0); rng = random.Random(0); R = Romanizer(seed=5)
L = [json.loads(l) for l in open(A.lists, encoding="utf-8")]
L = [x for x in L if len(x["passages"]) >= 4]
rng.shuffle(L); nd = max(50, len(L) // 33); dev, train = L[:nd], L[nd:]
print(f"train queries {len(train)} dev {len(dev)}", flush=True)
tok = AutoTokenizer.from_pretrained(A.base)
model = AutoModelForSequenceClassification.from_pretrained(A.base, num_labels=1, torch_dtype=torch.float32).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=A.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 300) * max(0.0, 1 - s / A.steps))
def view(x, mode):
    if mode == "roman": return R.sentence(x["q"], x["lang"], 1.0)
    if mode == "mixed": return R.sentence(x["q"], x["lang"], 0.5)
    return x["q"]
def group(x, r):
    t = np.array(x["teacher"]); idx = list(range(len(t)))
    pos = [i for i in idx if x["labels"][i] == 1]; best = list(np.argsort(-t)[:4])
    keep = list(dict.fromkeys(pos[:4] + best)); rest = [i for i in idx if i not in keep]; r.shuffle(rest)
    return (keep + rest)[:A.g]
def scores(qs, ps):
    b = tok(qs, ps, padding=True, truncation="only_second", max_length=A.max_len, return_tensors="pt").to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(**b).logits[:, 0].float()
def loss_on(batch, modes):
    qs, ps, ts, sizes = [], [], [], []
    for x, m in zip(batch, modes):
        g = group(x, rng); q = view(x, m)
        qs += [q] * len(g); ps += [x["passages"][i] for i in g]; ts += [x["teacher"][i] for i in g]; sizes.append(len(g))
    s = scores(qs, ps); t = torch.tensor(ts, device="cuda"); kl, k = 0.0, 0
    for n in sizes:
        kl = kl + torch.nn.functional.kl_div(torch.log_softmax(s[k:k + n], 0), torch.softmax(t[k:k + n], 0), reduction="sum"); k += n
    return kl / len(sizes) + 0.1 * ((s - t) ** 2).mean(), s, t, sizes
step = 0; ck = f"{A.out}/ckpt.pt"
if os.path.exists(ck):
    c = torch.load(ck, map_location="cuda"); model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"]); step = c["step"]
    print("resumed at step", step, flush=True)
@torch.no_grad()
def dev_eval():
    model.eval(); out = {}
    for mode in ("native", "roman"):
        r0 = random.Random(1); kls, agree = [], []
        for i in range(0, len(dev), A.q):
            b = dev[i:i + A.q]; qs, ps, ts, sizes = [], [], [], []
            for x in b:
                g = list(range(min(len(x["passages"]), 40))); q = R.sentence(x["q"], x["lang"], 1.0) if mode == "roman" else x["q"]
                qs += [q] * len(g); ps += [x["passages"][j] for j in g]; ts += [x["teacher"][j] for j in g]; sizes.append(len(g))
            s = scores(qs, ps); t = torch.tensor(ts, device="cuda"); k = 0
            for n in sizes:
                kls.append(torch.nn.functional.kl_div(torch.log_softmax(s[k:k + n], 0), torch.softmax(t[k:k + n], 0), reduction="sum").item())
                agree.append(float(s[k:k + n].argmax() == t[k:k + n].argmax())); k += n
        out[mode] = (np.mean(kls), np.mean(agree))
    model.train(); return out
stop = min(A.steps, step + A.chunk_steps) if A.chunk_steps else A.steps
t0 = time.time(); model.train()
while step < stop:
    batch = rng.sample(train, A.q)
    modes = ["roman" if (u := rng.random()) < A.roman else "mixed" if u < A.roman + A.mixed else "native" for _ in batch]
    loss, *_ = loss_on(batch, modes)
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
    if step % 100 == 0 or step == 1: print(f"step {step}/{A.steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    if step % A.ckpt_every == 0 or step == A.steps:
        d = dev_eval(); print(f"step {step} dev KL/top1 native {d['native'][0]:.3f}/{d['native'][1]:.3f} roman {d['roman'][0]:.3f}/{d['roman'][1]:.3f}", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, ck)
if step < A.steps: print(f"chunk done at step {step}/{A.steps}; rerun to resume", flush=True); raise SystemExit(0)
model.save_pretrained(A.out); tok.save_pretrained(A.out)
json.dump(vars(A), open(f"{A.out}/train_meta.json", "w")); print("saved", A.out)
