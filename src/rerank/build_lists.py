"""Training candidate lists for the reranker: MIRACL TRAIN queries (hi, bn, te) → judged passages (qrels, both labels) + the
top-K passages that indic-embed-small retrieves from that language's MIRACL corpus (hard negatives / unjudged). Then the teacher
(bge-reranker-v2-m3) scores every (native query, passage) pair. Dev queries are never used; any train query whose text equals a
MIRACLReranking dev query is dropped. python build_lists.py OUT EMBED_DIR TEACHER_DIR [K]
Output OUT/lists.jsonl: {"lang","qid","q","pids","passages","labels" (1/0/-1 unjudged),"teacher"}"""
import glob, gzip, json, os, sys, numpy as np, torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
out, emb_dir, tdir = sys.argv[1:4]; K = int(sys.argv[4]) if len(sys.argv) > 4 else 30; os.makedirs(out, exist_ok=True)
D = os.path.expanduser("~/Downloads/data")
dev_q = set()
for l in ["hi", "bn", "te"]:
    dev_q |= {r["text"].strip() for r in pq.read_table(glob.glob(f"{D}/MIRACLReranking/{l}-queries/*.parquet")[0]).to_pylist()}
    for f in glob.glob(f"{D}/miracl/miracl-v1.0-{l}/topics/*dev*.tsv"):
        dev_q |= {x.split("\t")[1].strip() for x in open(f, encoding="utf-8") if "\t" in x}
tok = AutoTokenizer.from_pretrained(emb_dir); enc = AutoModel.from_pretrained(emb_dir, torch_dtype=torch.bfloat16).cuda().eval()
sd = torch.load(f"{emb_dir}/head.pt", map_location="cuda"); head = torch.nn.Linear(sd["weight"].shape[1], sd["weight"].shape[0]).cuda(); head.load_state_dict(sd)
@torch.no_grad()
def embed(texts, bs=256, ml=256):
    out = []
    order = np.argsort([len(t) for t in texts]); inv = np.empty_like(order); inv[order] = np.arange(len(order))
    for i in range(0, len(texts), bs):
        b = tok([texts[j] for j in order[i:i + bs]], padding=True, truncation=True, max_length=ml, return_tensors="pt").to("cuda")
        h = enc(**b).last_hidden_state; m = b["attention_mask"][..., None]
        out.append(torch.nn.functional.normalize(head(((h * m).sum(1) / m.sum(1)).float()), dim=-1).half())
    return torch.cat(out)[torch.as_tensor(inv, device="cuda")]
ttok = AutoTokenizer.from_pretrained(tdir); tm = AutoModelForSequenceClassification.from_pretrained(tdir, torch_dtype=torch.bfloat16).cuda().eval()
@torch.no_grad()
def tscore(q, ps):
    s = []
    for i in range(0, len(ps), 64):
        b = ttok([q] * len(ps[i:i + 64]), ps[i:i + 64], padding=True, truncation="only_second", max_length=384, return_tensors="pt").to("cuda")
        s += tm(**b).logits[:, 0].float().tolist()
    return s
f = open(f"{out}/lists.jsonl", "w", encoding="utf-8")
for l in ["hi", "bn", "te"]:
    docs = {}
    for g in sorted(glob.glob(f"{D}/miracl-corpus/miracl-corpus-v1.0-{l}/*.jsonl.gz")):
        for line in gzip.open(g, "rt", encoding="utf-8"):
            x = json.loads(line); docs[x["docid"]] = (x["title"] + " " if x["title"] else "") + x["text"]
    ids = list(docs); print(l, "corpus", len(ids), flush=True)
    C = embed([docs[i] for i in ids])
    topics = {x.split("\t")[0]: x.split("\t")[1].strip() for x in open(glob.glob(f"{D}/miracl/miracl-v1.0-{l}/topics/*train.tsv")[0], encoding="utf-8") if "\t" in x}
    qrels = {}
    for x in open(glob.glob(f"{D}/miracl/miracl-v1.0-{l}/qrels/*train.tsv")[0], encoding="utf-8"):
        qid, _, pid, lab = x.split(); qrels.setdefault(qid, {})[pid] = int(lab)
    qids = [q for q in topics if q in qrels and topics[q] not in dev_q]; print(l, "train queries", len(qids), "dropped(dev overlap)", len([q for q in topics if topics[q] in dev_q]), flush=True)
    Q = embed([topics[q] for q in qids], ml=64)
    for i in range(0, len(qids), 512):
        top = (Q[i:i + 512] @ C.T).topk(K, dim=1).indices.tolist()
        for qid, t in zip(qids[i:i + 512], top):
            pids = list(qrels[qid]) + [ids[j] for j in t if ids[j] not in qrels[qid]]
            ps = [docs[p] for p in pids if p in docs]; pids = [p for p in pids if p in docs]
            f.write(json.dumps({"lang": l, "qid": qid, "q": topics[qid], "pids": pids, "passages": ps,
                                "labels": [qrels[qid].get(p, -1) for p in pids], "teacher": tscore(topics[qid], ps)}, ensure_ascii=False) + "\n")
        print(l, i + 512, flush=True)
    del C, docs; torch.cuda.empty_cache()
f.close(); print("done")
