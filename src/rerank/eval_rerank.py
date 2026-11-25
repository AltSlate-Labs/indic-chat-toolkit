"""Reranking evaluation on mteb/MIRACLReranking dev (MIRACL dev queries, 100 candidates each, CC BY-SA 4.0; MIRACL Apache-2.0).
python eval_rerank.py MODEL...   MODEL = name=dir_or_repo[:kind]   kind: ce (sequence-classification cross-encoder, default),
order (no reranking: candidates in the given first-stage order).
Tasks: nDCG@10 for hi, bn, te, en; and hi/bn/te with the query romanized by src/embed/romanizer.py (seed 11; synthetic
romanization, favours models trained on that romanizer's output). Passage = title + text."""
import ast, glob, math, os, sys, numpy as np, torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForSequenceClassification
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "embed"))
from romanizer import Romanizer
D = os.path.expanduser("~/Downloads/data/MIRACLReranking")
N = int(os.environ.get("RERANK_N", "350"))
def load(l):
    rd = lambda d: pq.read_table(glob.glob(f"{D}/{l}-{d}/*.parquet")[0]).to_pylist()
    corpus = {r["_id"]: ((r["title"] + " " if r["title"] else "") + r["text"]) for r in rd("corpus")}
    q = {r["_id"]: r["text"] for r in rd("queries")}; rel = {}
    for r in rd("qrels"): rel.setdefault(r["query-id"], {})[r["corpus-id"]] = int(float(r["score"]))
    tops = [(r["query-id"], ast.literal_eval(r["corpus-ids"]) if isinstance(r["corpus-ids"], str) else list(r["corpus-ids"])) for r in rd("top_ranked")][:N]
    return [(q[qid], [corpus[c] for c in cs], [rel.get(qid, {}).get(c, 0) for c in cs]) for qid, cs in tops]
R = Romanizer(seed=11)
DATA = {l: load(l) for l in ["hi", "bn", "te", "en"]}
for l in ["hi", "bn", "te"]:
    DATA[l + "-ro"] = [(R.sentence(q, l), p, g) for q, p, g in DATA[l]]
def ndcg10(scores, gains):
    order = np.argsort(-np.asarray(scores))[:10]; dcg = sum(gains[i] / math.log2(k + 2) for k, i in enumerate(order))
    ideal = sorted(gains, reverse=True)[:10]; idcg = sum(g / math.log2(k + 2) for k, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else None
class CE:
    def __init__(s, path):
        s.tok = AutoTokenizer.from_pretrained(path); s.m = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
    @torch.no_grad()
    def __call__(s, q, ps):
        out = []
        for i in range(0, len(ps), 50):
            b = s.tok([q] * len(ps[i:i + 50]), ps[i:i + 50], padding=True, truncation="only_second", max_length=384, return_tensors="pt").to("cuda")
            out += s.m(**b).logits[:, 0].float().tolist()
        return out
for spec in sys.argv[1:]:
    name, rest = spec.split("=", 1); path, kind = (rest.rsplit(":", 1) + ["ce"])[:2] if ":" in rest and rest.rsplit(":", 1)[1] in ("ce", "order") else (rest, "ce")
    model = CE(path) if kind == "ce" else None; res = {}
    for task, rows in DATA.items():
        v = [ndcg10(model(q, ps) if model else list(range(len(ps), 0, -1)), g) for q, ps, g in rows]
        res[task] = float(np.mean([x for x in v if x is not None]))
    nat = np.mean([res[l] for l in ["hi", "bn", "te"]]); ro = np.mean([res[l + "-ro"] for l in ["hi", "bn", "te"]])
    print(f"{name:28s} native {nat:.3f} | roman {ro:.3f} | " + " ".join(f"{t}:{x:.3f}" for t, x in res.items()), flush=True)
    del model; torch.cuda.empty_cache()
