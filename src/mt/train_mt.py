"""Fine-tune a seq2seq model (Marian / mT5 / ByT5 / M2M100 with --src_lang) on jsonl rows {src, tgt}; checkpoints, then decodes the dev split and prints
chrF++/BLEU (lower-cased, like eval_mt.py). python train_mt.py --base DIR --data TRAIN_JSONL --dev DEV_JSONL --out DIR --steps N"""
import argparse, json, os, random, time, torch, sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
p = argparse.ArgumentParser()
for a, d in [("--base", None), ("--data", None), ("--dev", None), ("--out", None)]: p.add_argument(a, default=d)
p.add_argument("--steps", type=int, default=3000); p.add_argument("--bs", type=int, default=64); p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--max_len", type=int, default=128); p.add_argument("--ckpt_every", type=int, default=2000); p.add_argument("--dev_n", type=int, default=1000)
p.add_argument("--prefix", default=""); p.add_argument("--src_lang", default="")   # M2M100: source code (e.g. hi), or "auto" = from kind suffix (xl-ta, xl-roman-ta) else hi
A = p.parse_args(); os.makedirs(A.out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(A.base); m = AutoModelForSeq2SeqLM.from_pretrained(A.base).cuda()
GK = {}
if A.src_lang: tok.src_lang, tok.tgt_lang = ("hi" if A.src_lang == "auto" else A.src_lang), "en"; GK = {"forced_bos_token_id": tok.get_lang_id("en")}
# M2M100 vocab lacks Telugu script (41% <unk>): map Telugu letters to Kannada (same layout, Unicode block +0x80)
te2kn = lambda t: "".join(chr(ord(c) + 0x80) if 0xC00 <= ord(c) <= 0xC7F else c for c in t)
LANGS = {"hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "ur"}
def lang_of(r):
    k = r.get("kind", "").rsplit("-", 1)[-1]
    l = r.get("lang") or (k if k in LANGS else "hi")
    return "kn" if l == "te" else l   # M2M100 has no Telugu code; Kannada is the closest (script and family)
def enc(b):
    if A.src_lang != "auto":
        x = tok([A.prefix + r["src"] for r in b], text_target=[r["tgt"] for r in b], padding=True, truncation=True, max_length=A.max_len, return_tensors="pt")
    else:   # per-row source-language token: tokenize each language group, then pad together
        ids = [None] * len(b)
        for lg in {lang_of(r) for r in b}:
            ix = [i for i, r in enumerate(b) if lang_of(r) == lg]; tok.src_lang = lg
            for i, e in zip(ix, tok([te2kn(b[i]["src"]) for i in ix], truncation=True, max_length=A.max_len)["input_ids"]): ids[i] = e
        tok.src_lang = "hi"
        x = tok.pad({"input_ids": ids}, return_tensors="pt")
        x["labels"] = tok.pad({"input_ids": tok(text_target=[r["tgt"] for r in b], truncation=True, max_length=A.max_len)["input_ids"]}, return_tensors="pt")["input_ids"]
    return x
print("params", sum(x.numel() for x in m.parameters()) // 10**6, "M")
rows = [json.loads(l) for l in open(A.data, encoding="utf-8")]; print("train rows", len(rows))
opt = torch.optim.AdamW(m.parameters(), lr=A.lr, weight_decay=0.01); sch = get_linear_schedule_with_warmup(opt, min(500, A.steps // 10), A.steps)
step, t0, rng = 0, time.time(), random.Random(1)
ck = f"{A.out}/ckpt.pt"
if os.path.exists(ck):
    s = torch.load(ck, map_location="cuda"); m.load_state_dict(s["m"]); opt.load_state_dict(s["o"]); sch.load_state_dict(s["s"]); step = s["step"]; print("resumed", step)
m.train()
while step < A.steps:
    b = rng.sample(rows, A.bs)
    x = enc(b).to("cuda")
    x["labels"][x["labels"] == tok.pad_token_id] = -100
    with torch.autocast("cuda", dtype=torch.bfloat16): loss = m(**x).loss
    loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(); step += 1
    if step % 200 == 0: print(f"step {step}/{A.steps} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
    if step % A.ckpt_every == 0 and step < A.steps: torch.save({"m": m.state_dict(), "o": opt.state_dict(), "s": sch.state_dict(), "step": step}, ck)
m.save_pretrained(f"{A.out}/model"); tok.save_pretrained(f"{A.out}/model")
if os.path.exists(ck): os.remove(ck)
m.eval(); dev = [json.loads(l) for l in open(A.dev, encoding="utf-8")][:A.dev_n]; hyp = []
for i in range(0, len(dev), 64):
    x = tok([A.prefix + r["src"] for r in dev[i:i + 64]], return_tensors="pt", padding=True, truncation=True, max_length=A.max_len).to("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): o = m.generate(**x, num_beams=4, max_new_tokens=160, **GK)
    hyp += tok.batch_decode(o, skip_special_tokens=True)
with open(f"{A.out}/pred-dev.jsonl", "w", encoding="utf-8") as f:
    for r, h in zip(dev, hyp): f.write(json.dumps({"src": r["src"], "hyp": h, "ref": r["ref"]}, ensure_ascii=False) + "\n")
L = lambda xs: [x.lower() for x in xs]; ref = [r["ref"] for r in dev]
print(f"DEV {os.path.basename(A.out)} chrF++ {sacrebleu.corpus_chrf(L(hyp), [L(ref)], word_order=2).score:.1f} BLEU {sacrebleu.corpus_bleu(L(hyp), [L(ref)]).score:.1f}")
json.dump(vars(A), open(f"{A.out}/train_meta.json", "w"))
