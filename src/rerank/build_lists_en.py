"""English candidate lists for the reranker from sentence-transformers/natural-questions (pairs derived from Google's Natural
Questions, CC BY-SA 3.0): N questions; candidates = the gold answer passage + the top-K passages that indic-embed-small retrieves
among all ~100k answer passages; teacher (bge-reranker-v2-m3) scores each pair. Questions whose text equals a MIRACLReranking
English dev query are dropped. python build_lists_en.py PARQUET OUT EMBED_DIR TEACHER_DIR [N] [K]   (same format as build_lists.py)"""
import glob, json, os, random, sys, numpy as np, torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
src, out, emb_dir, tdir = sys.argv[1:5]; N = int(sys.argv[5]) if len(sys.argv) > 5 else 6000; K = int(sys.argv[6]) if len(sys.argv) > 6 else 30
os.makedirs(out, exist_ok=True); D = os.path.expanduser("~/Downloads/data")
dev_q = {r["text"].strip().lower() for r in pq.read_table(glob.glob(f"{D}/MIRACLReranking/en-queries/*.parquet")[0]).to_pylist()}
rows = pq.read_table(src).to_pylist(); qcol, acol = list(rows[0].keys())[:2]
pas = sorted({r[acol] for r in rows}); pidx = {p: i for i, p in enumerate(pas)}
cand = [r for r in rows if r[qcol].strip().lower() not in dev_q]; print("pairs", len(rows), "passages", len(pas), "dropped(dev)", len(rows) - len(cand), flush=True)
random.Random(0).shuffle(cand); cand = cand[:N]
tok = AutoTokenizer.from_pretrained(emb_dir); enc = AutoModel.from_pretrained(emb_dir, torch_dtype=torch.bfloat16).cuda().eval()
sd = torch.load(f"{emb_dir}/head.pt", map_location="cuda"); head = torch.nn.Linear(sd["weight"].shape[1], sd["weight"].shape[0]).cuda(); head.load_state_dict(sd)
@torch.no_grad()
def embed(texts, bs=256, ml=256):
    order = np.argsort([len(t) for t in texts]); inv = np.empty_like(order); inv[order] = np.arange(len(order)); out = []
    for i in range(0, len(texts), bs):
        b = tok([texts[j] for j in order[i:i + bs]], padding=True, truncation=True, max_length=ml, return_tensors="pt").to("cuda")
        h = enc(**b).last_hidden_state; m = b["attention_mask"][..., None]
        out.append(torch.nn.functional.normalize(head(((h * m).sum(1) / m.sum(1)).float()), dim=-1).half())
    return torch.cat(out)[torch.as_tensor(inv, device="cuda")]
C = embed(pas); Q = embed([r[qcol] for r in cand], ml=64)
ttok = AutoTokenizer.from_pretrained(tdir); tm = AutoModelForSequenceClassification.from_pretrained(tdir, torch_dtype=torch.bfloat16).cuda().eval()
@torch.no_grad()
def tscore(q, ps):
    s = []
    for i in range(0, len(ps), 64):
        b = ttok([q] * len(ps[i:i + 64]), ps[i:i + 64], padding=True, truncation="only_second", max_length=384, return_tensors="pt").to("cuda")
        s += tm(**b).logits[:, 0].float().tolist()
    return s
with open(f"{out}/lists.jsonl", "w", encoding="utf-8") as f:
    for i in range(0, len(cand), 512):
        top = (Q[i:i + 512] @ C.T).topk(K, dim=1).indices.tolist()
        for r, t in zip(cand[i:i + 512], top):
            g = pidx[r[acol]]; ids = [g] + [j for j in t if j != g]; ps = [pas[j] for j in ids]
            f.write(json.dumps({"lang": "en", "qid": f"nq{i}", "q": r[qcol], "pids": [str(j) for j in ids], "passages": ps,
                                "labels": [1] + [-1] * (len(ids) - 1), "teacher": tscore(r[qcol], ps)}, ensure_ascii=False) + "\n")
        print("en", i + 512, flush=True)
print("done")
